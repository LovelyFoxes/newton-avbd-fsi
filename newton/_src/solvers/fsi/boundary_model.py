"""Boundary sample model for AVBD/IPBF fluid-solid coupling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...sim import Model, State
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
        shape_indices: Optional subset of shape indices to sample.
        include_static: Whether static shapes with ``shape_body == -1`` are sampled.
        include_dynamic: Whether body-attached shapes are sampled.
        device: Warp device. Defaults to ``model.device``.

    Returns:
        Boundary sample model.
    """

    def __init__(
        self,
        model: Model,
        spacing: float,
        *,
        support_radius: float | None = None,
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
        self.device = wp.get_device(device if device is not None else model.device)

        sample_body, sample_shape, sample_x_local, sample_normal_local, sample_volume = self._sample_model_boxes(
            model,
            self.spacing,
            shape_indices=shape_indices,
            include_static=include_static,
            include_dynamic=include_dynamic,
        )

        self.sample_count = len(sample_body)
        self.sample_body = wp.array(sample_body, dtype=wp.int32, device=self.device)
        self.sample_shape = wp.array(sample_shape, dtype=wp.int32, device=self.device)
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
        self.sample_mass_equiv = wp.zeros(self.sample_count, dtype=float, device=self.device)

        self.sample_x_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_v_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_normal_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_force = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)

        body_count = int(getattr(model, "body_count", 0))
        self.body_force = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_torque = wp.zeros(body_count, dtype=wp.vec3, device=self.device)

        self.boundary_grid = wp.HashGrid(128, 128, 128) if self.sample_count > 0 else None

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
    ) -> tuple[list[int], list[int], list[tuple[float, float, float]], list[tuple[float, float, float]], list[float]]:
        if model.shape_count == 0:
            return [], [], [], [], []

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
        sample_volume: list[float] = []

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
                sample_volume.append(spacing * spacing * spacing)

        return sample_body, sample_shape, sample_x_local, sample_normal_local, sample_volume

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
