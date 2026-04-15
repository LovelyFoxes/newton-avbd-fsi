# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Boundary sample model for AVBD/IPBF fluid-solid coupling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...sim import BodyFlags, Model, State
from .boundary_kernels import update_boundary_sample_world_kinematics

__all__ = [
    "BoundarySampleFlags",
    "FSIBoundaryModel",
]


class BoundarySampleFlags(IntEnum):
    """Flags describing a fluid-solid boundary sample."""

    ACTIVE = 1 << 0
    STATIC = 1 << 1
    DYNAMIC = 1 << 2


class FSIBoundaryModel:
    """Boundary samples used by the AVBD/IPBF fluid-solid coupling path.

    The first implementation follows an Akinci-style boundary particle model
    and only samples Newton box shapes. Samples are stored in the parent frame:
    body-local for dynamic shapes and world-space for static shapes.

    Args:
        model: Newton model containing shapes to sample.
        spacing: Boundary sample spacing [m].
        support_radius: Neighbor query support radius [m]. Defaults to
            ``spacing`` until IPBF density kernels are wired in.
        kernel_family: SPH kernel used to calibrate Akinci-style sample
            volumes. Defaults to cubic spline and should usually match the
            fluid solver kernel family.
        hydrostatic_volume_mode: Optional hydrostatic normalization applied to
            boundary sample volumes before they are consumed by density and
            pressure-coupling paths.
        shape_indices: Optional subset of shape indices to sample.
        include_static: Whether static shapes with ``shape_body == -1`` are sampled.
        include_dynamic: Whether body-attached shapes are sampled.
        device: Warp device. Defaults to ``model.device``.

    Returns:
        Boundary sample model.
    """

    class KernelFamily(IntEnum):
        """Selectable SPH kernels used for boundary-volume calibration."""

        CUBIC_SPLINE = 0
        POLY6 = 1

    class HydrostaticVolumeMode(IntEnum):
        """Selectable hydrostatic-normalization modes for boundary shells."""

        NONE = 0
        DYNAMIC_BOX_VOLUME = 1

    def __init__(
        self,
        model: Model,
        spacing: float,
        *,
        support_radius: float | None = None,
        kernel_family: KernelFamily = KernelFamily.CUBIC_SPLINE,
        hydrostatic_volume_mode: HydrostaticVolumeMode = HydrostaticVolumeMode.NONE,
        shape_indices: Sequence[int] | None = None,
        include_static: bool = True,
        include_dynamic: bool = True,
        device: wp.context.Devicelike | None = None,
    ):
        if spacing <= 0.0:
            raise ValueError("Boundary sample spacing must be positive.")

        self.model = model
        self.spacing = float(spacing)
        self.support_radius = float(support_radius) if support_radius is not None else float(spacing)
        if self.support_radius <= 0.0:
            raise ValueError("Boundary sample support radius must be positive.")

        self.kernel_family = int(self.KernelFamily(kernel_family))
        self.hydrostatic_volume_mode = int(self.HydrostaticVolumeMode(hydrostatic_volume_mode))
        self.device = wp.get_device(device if device is not None else model.device)

        sample_body, sample_shape, sample_x_local, sample_normal_local = self._sample_model_boxes(
            model,
            self.spacing,
            shape_indices=shape_indices,
            include_static=include_static,
            include_dynamic=include_dynamic,
        )
        sample_volume = self._calibrate_sample_volumes(
            sample_body,
            sample_x_local,
            self.support_radius,
            self.kernel_family,
        )
        sample_volume_hydrostatic, sample_volume_hydrostatic_scale = self._compute_hydrostatic_sample_volumes(
            model,
            sample_shape,
            sample_volume,
            self.hydrostatic_volume_mode,
        )

        self.sample_count = len(sample_body)
        shape_sample_count = [0 for _ in range(model.shape_count)]
        for shape in sample_shape:
            shape_sample_count[shape] += 1

        self.sample_body = wp.array(sample_body, dtype=wp.int32, device=self.device)
        self.sample_shape = wp.array(sample_shape, dtype=wp.int32, device=self.device)
        self.shape_sample_count = wp.array(shape_sample_count, dtype=wp.int32, device=self.device)
        self.sample_flags = wp.array(
            [
                int(BoundarySampleFlags.ACTIVE)
                | (int(BoundarySampleFlags.DYNAMIC) if body >= 0 else int(BoundarySampleFlags.STATIC))
                for body in sample_body
            ],
            dtype=wp.int32,
            device=self.device,
        )
        self.sample_x_local = wp.array(sample_x_local, dtype=wp.vec3, device=self.device)
        self.sample_normal_local = wp.array(sample_normal_local, dtype=wp.vec3, device=self.device)
        self.sample_volume = wp.array(sample_volume, dtype=float, device=self.device)
        self.sample_volume_hydrostatic = wp.array(sample_volume_hydrostatic, dtype=float, device=self.device)
        self.sample_volume_hydrostatic_scale = wp.array(
            sample_volume_hydrostatic_scale,
            dtype=float,
            device=self.device,
        )
        self.sample_mass_equiv = wp.zeros(self.sample_count, dtype=float, device=self.device)

        self.sample_x_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_v_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_normal_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_force = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)

        body_count = int(getattr(model, "body_count", 0))
        self.body_force = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_torque = wp.zeros(body_count, dtype=wp.vec3, device=self.device)

        self.boundary_grid = wp.HashGrid(128, 128, 128, device=self.device) if self.sample_count > 0 else None

        self._empty_body_q = wp.zeros(0, dtype=wp.transform, device=self.device)
        self._empty_body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=self.device)
        self._empty_body_com = wp.zeros(0, dtype=wp.vec3, device=self.device)

    def update_world_kinematics(self, state: State) -> None:
        """Update sample positions [m], velocities [m/s], and normals."""
        if self.sample_count == 0:
            return

        wp.launch(
            update_boundary_sample_world_kinematics,
            dim=self.sample_count,
            inputs=[
                self.sample_body,
                self.sample_x_local,
                self.sample_normal_local,
                state.body_q if state.body_q is not None else self._empty_body_q,
                state.body_qd if state.body_qd is not None else self._empty_body_qd,
                self.model.body_com if self.model.body_com is not None else self._empty_body_com,
            ],
            outputs=[
                self.sample_x_world,
                self.sample_v_world,
                self.sample_normal_world,
            ],
            device=self.device,
        )

    def build_grid(self) -> None:
        """Build the boundary sample hash grid for neighbor queries."""
        if self.boundary_grid is None or self.sample_count == 0:
            return

        with wp.ScopedDevice(self.device):
            self.boundary_grid.build(self.sample_x_world, radius=self.support_radius)

    def clear_forces(self) -> None:
        """Clear boundary sample and body wrench accumulators."""
        self.sample_force.zero_()
        self.body_force.zero_()
        self.body_torque.zero_()

    @classmethod
    def _sample_model_boxes(
        cls,
        model: Model,
        spacing: float,
        *,
        shape_indices: Sequence[int] | None,
        include_static: bool,
        include_dynamic: bool,
    ) -> tuple[list[int], list[int], list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        if model.shape_count == 0:
            return [], [], [], []

        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_transform = model.shape_transform.numpy()

        if shape_indices is None:
            indices = range(model.shape_count)
        else:
            indices = shape_indices

        sample_body: list[int] = []
        sample_shape: list[int] = []
        sample_x_local: list[tuple[float, float, float]] = []
        sample_normal_local: list[tuple[float, float, float]] = []

        for shape_idx in indices:
            if shape_idx < 0 or shape_idx >= model.shape_count:
                raise ValueError(f"Shape index {shape_idx} is out of bounds for {model.shape_count} shapes.")

            if int(shape_type[shape_idx]) != int(GeoType.BOX):
                continue

            body = int(shape_body[shape_idx])
            if body < 0 and not include_static:
                continue
            if body >= 0 and not include_dynamic:
                continue

            X_parent_shape = wp.transform(*shape_transform[shape_idx])
            hx = float(shape_scale[shape_idx][0])
            hy = float(shape_scale[shape_idx][1])
            hz = float(shape_scale[shape_idx][2])

            for point_shape, normal_shape in cls._sample_box_surface(hx, hy, hz, spacing):
                point_parent = wp.transform_point(X_parent_shape, wp.vec3(*point_shape))
                normal_parent = wp.normalize(wp.transform_vector(X_parent_shape, wp.vec3(*normal_shape)))

                sample_body.append(body)
                sample_shape.append(int(shape_idx))
                sample_x_local.append(cls._vec3_to_tuple(point_parent))
                sample_normal_local.append(cls._vec3_to_tuple(normal_parent))

        return sample_body, sample_shape, sample_x_local, sample_normal_local

    @classmethod
    def _calibrate_sample_volumes(
        cls,
        sample_body: Sequence[int],
        sample_x_local: Sequence[tuple[float, float, float]],
        support_radius: float,
        kernel_family: int,
    ) -> list[float]:
        if len(sample_x_local) == 0:
            return []

        positions = np.asarray(sample_x_local, dtype=np.float32)
        volumes = np.zeros(len(sample_x_local), dtype=np.float32)
        group_indices: dict[tuple[str, int], list[int]] = {}

        for sample_index, body in enumerate(sample_body):
            # Dynamic samples live in body-local frames, so calibrate them
            # per body to avoid unrelated local frames overlapping.
            group_key = ("static", 0) if body < 0 else ("body", int(body))
            group_indices.setdefault(group_key, []).append(sample_index)

        for group in group_indices.values():
            group_index_array = np.asarray(group, dtype=np.int32)
            group_positions = positions[group_index_array]
            deltas = cls._compute_boundary_volume_deltas(group_positions, support_radius, kernel_family)
            volumes[group_index_array] = np.where(deltas > 0.0, 1.0 / deltas, 0.0).astype(np.float32)

        return volumes.tolist()

    @classmethod
    def _compute_boundary_volume_deltas(
        cls,
        positions: np.ndarray,
        support_radius: float,
        kernel_family: int,
    ) -> np.ndarray:
        if len(positions) == 0:
            return np.zeros(0, dtype=np.float32)

        cell_size = support_radius
        cell_indices = np.floor(positions / cell_size).astype(np.int32)
        cells: dict[tuple[int, int, int], list[int]] = {}
        for sample_index, cell_index in enumerate(cell_indices):
            cell_key = (int(cell_index[0]), int(cell_index[1]), int(cell_index[2]))
            cells.setdefault(cell_key, []).append(sample_index)

        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        delta = np.full(len(positions), cls._kernel_value_numpy(0.0, support_radius, kernel_family), dtype=np.float64)

        for sample_index, position in enumerate(positions):
            base_cell = cell_indices[sample_index]
            for dx, dy, dz in offsets:
                cell_key = (int(base_cell[0] + dx), int(base_cell[1] + dy), int(base_cell[2] + dz))
                candidate_indices = cells.get(cell_key)
                if candidate_indices is None:
                    continue

                displacement = positions[candidate_indices] - position
                dist2 = np.einsum("ij,ij->i", displacement, displacement, dtype=np.float64)
                neighbor_mask = dist2 > 0.0
                if not np.any(neighbor_mask):
                    continue

                delta[sample_index] += np.sum(
                    cls._kernel_value_numpy(dist2[neighbor_mask], support_radius, kernel_family)
                )

        return delta.astype(np.float32)

    @classmethod
    def _compute_hydrostatic_sample_volumes(
        cls,
        model: Model,
        sample_shape: Sequence[int],
        sample_volume: Sequence[float],
        hydrostatic_volume_mode: int,
    ) -> tuple[list[float], list[float]]:
        if len(sample_volume) == 0:
            return [], []

        volume_hydrostatic = np.asarray(sample_volume, dtype=np.float32).copy()
        hydrostatic_scale = np.ones(len(sample_volume), dtype=np.float32)

        if hydrostatic_volume_mode == int(cls.HydrostaticVolumeMode.NONE):
            return volume_hydrostatic.tolist(), hydrostatic_scale.tolist()

        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        body_flags = model.body_flags.numpy() if model.body_flags is not None else None

        sample_indices_by_shape: dict[int, list[int]] = {}
        for sample_index, shape_index in enumerate(sample_shape):
            sample_indices_by_shape.setdefault(int(shape_index), []).append(sample_index)

        for shape_index, indices in sample_indices_by_shape.items():
            if int(shape_type[shape_index]) != int(GeoType.BOX):
                continue

            body = int(shape_body[shape_index])
            if body < 0:
                continue

            if hydrostatic_volume_mode == int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_VOLUME):
                if body_flags is None or (int(body_flags[body]) & int(BodyFlags.KINEMATIC)) != 0:
                    continue

            hx = float(shape_scale[shape_index][0])
            hy = float(shape_scale[shape_index][1])
            hz = float(shape_scale[shape_index][2])
            target_volume = 8.0 * hx * hy * hz
            if target_volume <= 0.0:
                continue

            raw_sum = float(np.sum(volume_hydrostatic[indices]))
            if raw_sum <= 0.0:
                continue

            scale = target_volume / raw_sum
            volume_hydrostatic[indices] *= scale
            hydrostatic_scale[indices] = scale

        return volume_hydrostatic.tolist(), hydrostatic_scale.tolist()

    @classmethod
    def _kernel_value_numpy(
        cls,
        dist2: float | np.ndarray,
        support_radius: float,
        kernel_family: int,
    ) -> float | np.ndarray:
        if support_radius <= 0.0:
            if np.ndim(dist2) == 0:
                return 0.0
            return np.zeros_like(np.asarray(dist2, dtype=np.float64), dtype=np.float64)

        dist2_array = np.asarray(dist2, dtype=np.float64)
        values = np.zeros_like(dist2_array, dtype=np.float64)

        if kernel_family == int(cls.KernelFamily.POLY6):
            support_radius2 = support_radius * support_radius
            inside = dist2_array < support_radius2
            x = support_radius2 - dist2_array[inside]
            values[inside] = 315.0 / (64.0 * np.pi * support_radius**9) * x * x * x
        else:
            distances = np.sqrt(dist2_array)
            inside = distances < support_radius
            q = 2.0 * distances[inside] / support_radius
            normalization = 8.0 / (np.pi * support_radius**3)
            inside_q = q < 1.0

            values_inside = np.empty_like(q)
            values_inside[inside_q] = normalization * (1.0 - 1.5 * q[inside_q] ** 2 + 0.75 * q[inside_q] ** 3)
            two_minus_q = 2.0 - q[~inside_q]
            values_inside[~inside_q] = normalization * 0.25 * two_minus_q * two_minus_q * two_minus_q
            values[inside] = values_inside

        if values.ndim == 0:
            return float(values)

        return values

    @staticmethod
    def _sample_box_surface(
        hx: float, hy: float, hz: float, spacing: float
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        xs = FSIBoundaryModel._axis_samples(hx, spacing)
        ys = FSIBoundaryModel._axis_samples(hy, spacing)
        zs = FSIBoundaryModel._axis_samples(hz, spacing)

        samples: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for ix, x in enumerate(xs):
            nx = FSIBoundaryModel._surface_axis_normal(ix, len(xs))
            for iy, y in enumerate(ys):
                ny = FSIBoundaryModel._surface_axis_normal(iy, len(ys))
                for iz, z in enumerate(zs):
                    nz = FSIBoundaryModel._surface_axis_normal(iz, len(zs))
                    if nx == 0.0 and ny == 0.0 and nz == 0.0:
                        continue

                    normal = np.array([nx, ny, nz], dtype=np.float32)
                    normal /= np.linalg.norm(normal)
                    samples.append(((float(x), float(y), float(z)), tuple(float(v) for v in normal)))

        return samples

    @staticmethod
    def _axis_samples(half_extent: float, spacing: float) -> np.ndarray:
        if half_extent <= 0.0:
            return np.array([0.0], dtype=np.float32)

        segments = max(1, int(math.ceil((2.0 * half_extent) / spacing)))
        return np.linspace(-half_extent, half_extent, segments + 1, dtype=np.float32)

    @staticmethod
    def _surface_axis_normal(index: int, count: int) -> float:
        if count <= 1:
            return 0.0
        if index == 0:
            return -1.0
        if index == count - 1:
            return 1.0
        return 0.0

    @staticmethod
    def _vec3_to_tuple(value: wp.vec3) -> tuple[float, float, float]:
        return (float(value[0]), float(value[1]), float(value[2]))
