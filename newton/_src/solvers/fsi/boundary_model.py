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
from .boundary_kernels import (
    accumulate_step_reaction_diagnostics,
    update_boundary_sample_world_kinematics,
    update_deformable_boundary_sample_hydrostatic_state,
    update_deformable_boundary_sample_world_kinematics,
    update_deformable_triangle_contact_proxy_query_kinematics,
    update_deformable_triangle_contact_proxy_world_kinematics,
    update_deformable_triangle_contact_swept_aabbs,
    update_step_reaction_averages,
)

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

    The implementation follows an Akinci-style boundary particle model and
    samples Newton box and sphere shapes, plus optional triangle elements.
    Samples are stored in the parent frame: body-local for dynamic shapes and
    world-space for static shapes.

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
        Step-level reaction diagnostics are accumulated separately from the
            transient ``body_force`` and ``body_torque`` buffers so interlinked
            solver schedules can report the average wrench consumed over a
            timestep.
        shape_indices: Optional subset of shape indices to sample.
        include_triangles: Whether triangle elements are sampled as deformable
            thin-shell boundary samples.
        triangle_indices: Optional subset of triangle indices to sample.
        deformable_sample_thickness: Effective deformable shell thickness [m]
            used to convert triangle patch areas into boundary sample volumes.
            Defaults to ``spacing``.
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
        DYNAMIC_SHAPE_VOLUME = 1
        DYNAMIC_SHAPE_SURFACE_QUADRATURE = 2
        DYNAMIC_SHAPE_SURFACE_THICKNESS = 3
        DYNAMIC_BOX_VOLUME = 1
        DYNAMIC_BOX_SURFACE_QUADRATURE = 2
        DYNAMIC_BOX_SURFACE_THICKNESS = 3

    def __init__(
        self,
        model: Model,
        spacing: float,
        *,
        support_radius: float | None = None,
        kernel_family: KernelFamily = KernelFamily.CUBIC_SPLINE,
        hydrostatic_volume_mode: HydrostaticVolumeMode = HydrostaticVolumeMode.NONE,
        shape_indices: Sequence[int] | None = None,
        include_triangles: bool = False,
        triangle_indices: Sequence[int] | None = None,
        deformable_sample_thickness: float | None = None,
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
        if deformable_sample_thickness is not None and deformable_sample_thickness <= 0.0:
            raise ValueError("Deformable boundary sample thickness must be positive.")
        self.deformable_sample_thickness = (
            float(deformable_sample_thickness) if deformable_sample_thickness is not None else self.spacing
        )
        self.triangle_contact_aabb_padding = max(1.0e-6, 0.25 * self.deformable_sample_thickness)

        self.kernel_family = int(self.KernelFamily(kernel_family))
        self.hydrostatic_volume_mode = int(self.HydrostaticVolumeMode(hydrostatic_volume_mode))
        self.device = wp.get_device(device if device is not None else model.device)

        (
            sample_body,
            sample_shape,
            sample_x_local,
            sample_normal_local,
            sample_area_patch,
            sample_volume_quadrature,
        ) = self._sample_model_shapes(
            model,
            self.spacing,
            shape_indices=shape_indices,
            include_static=include_static,
            include_dynamic=include_dynamic,
        )
        (
            sample_triangle,
            sample_vertex0,
            sample_vertex1,
            sample_vertex2,
            sample_barycentric,
            triangle_sample_body,
            triangle_sample_shape,
            triangle_sample_x_local,
            triangle_sample_normal_local,
            triangle_sample_offset_distance,
            triangle_sample_area_patch,
            triangle_sample_volume_quadrature,
            triangle_sample_rest_area,
        ) = self._sample_model_triangles(
            model,
            self.spacing,
            thickness=self.deformable_sample_thickness,
            triangle_indices=triangle_indices,
            include_triangles=include_triangles,
        )

        shape_sample_count_total = len(sample_body)
        sample_body.extend(triangle_sample_body)
        sample_shape.extend(triangle_sample_shape)
        sample_x_local.extend(triangle_sample_x_local)
        sample_normal_local.extend(triangle_sample_normal_local)
        sample_offset_distance = [0.0 for _ in range(shape_sample_count_total)]
        sample_offset_distance.extend(triangle_sample_offset_distance)
        sample_area_patch.extend(triangle_sample_area_patch)
        sample_volume_quadrature.extend(triangle_sample_volume_quadrature)
        sample_triangle_rest_area = [0.0 for _ in range(shape_sample_count_total)]
        sample_triangle_rest_area.extend(triangle_sample_rest_area)

        sample_volume = self._calibrate_sample_volumes(
            sample_body[:shape_sample_count_total],
            sample_x_local[:shape_sample_count_total],
            self.support_radius,
            self.kernel_family,
        )
        sample_volume.extend(sample_volume_quadrature[shape_sample_count_total:])
        sample_volume_hydrostatic, sample_volume_hydrostatic_scale = self._compute_hydrostatic_sample_volumes(
            model,
            sample_shape,
            sample_volume,
            sample_area_patch,
            sample_volume_quadrature,
            self.hydrostatic_volume_mode,
        )

        self.sample_count = len(sample_body)
        self.shape_sample_count_total = shape_sample_count_total
        self.triangle_sample_count = len(sample_triangle)
        self.triangle_indices = wp.array(sorted(set(sample_triangle)), dtype=wp.int32, device=self.device)
        self.triangle_count = self.triangle_indices.shape[0]
        triangle_contact_radius = self._compute_triangle_contact_radii(model, sorted(set(sample_triangle)))
        self.triangle_contact_radius = wp.array(triangle_contact_radius, dtype=float, device=self.device)
        self.triangle_contact_radius_max = max(triangle_contact_radius, default=0.0)
        shape_sample_count = [0 for _ in range(model.shape_count)]
        for shape in sample_shape:
            if shape >= 0:
                shape_sample_count[shape] += 1

        self.sample_body = wp.array(sample_body, dtype=wp.int32, device=self.device)
        self.sample_shape = wp.array(sample_shape, dtype=wp.int32, device=self.device)
        self.shape_sample_count = wp.array(shape_sample_count, dtype=wp.int32, device=self.device)
        self.sample_triangle = wp.array(
            [-1 for _ in range(shape_sample_count_total)] + sample_triangle,
            dtype=wp.int32,
            device=self.device,
        )
        self.sample_vertex0 = wp.array(
            [-1 for _ in range(shape_sample_count_total)] + sample_vertex0,
            dtype=wp.int32,
            device=self.device,
        )
        self.sample_vertex1 = wp.array(
            [-1 for _ in range(shape_sample_count_total)] + sample_vertex1,
            dtype=wp.int32,
            device=self.device,
        )
        self.sample_vertex2 = wp.array(
            [-1 for _ in range(shape_sample_count_total)] + sample_vertex2,
            dtype=wp.int32,
            device=self.device,
        )
        self.sample_barycentric = wp.array(
            [(0.0, 0.0, 0.0) for _ in range(shape_sample_count_total)] + sample_barycentric,
            dtype=wp.vec3,
            device=self.device,
        )
        sample_flags = []
        for sample_index, body in enumerate(sample_body):
            is_triangle_sample = sample_index >= shape_sample_count_total
            dynamic = body >= 0 or is_triangle_sample
            sample_flags.append(
                int(BoundarySampleFlags.ACTIVE)
                | (int(BoundarySampleFlags.DYNAMIC) if dynamic else int(BoundarySampleFlags.STATIC))
            )
        self.sample_flags = wp.array(
            sample_flags,
            dtype=wp.int32,
            device=self.device,
        )

        self.sample_x_local = wp.array(sample_x_local, dtype=wp.vec3, device=self.device)
        self.sample_normal_local = wp.array(sample_normal_local, dtype=wp.vec3, device=self.device)
        self.sample_offset_distance = wp.array(sample_offset_distance, dtype=float, device=self.device)
        self.sample_volume = wp.array(sample_volume, dtype=float, device=self.device)
        self.sample_volume_rest = wp.clone(self.sample_volume)
        self.sample_area_patch = wp.array(sample_area_patch, dtype=float, device=self.device)
        self.sample_area_patch_rest = wp.clone(self.sample_area_patch)
        self.sample_volume_quadrature = wp.array(sample_volume_quadrature, dtype=float, device=self.device)
        self.sample_volume_quadrature_rest = wp.clone(self.sample_volume_quadrature)
        self.sample_area_box_patch = self.sample_area_patch
        self.sample_volume_box_quadrature = self.sample_volume_quadrature
        self.sample_volume_hydrostatic = wp.array(sample_volume_hydrostatic, dtype=float, device=self.device)
        self.sample_volume_hydrostatic_rest = wp.clone(self.sample_volume_hydrostatic)
        self.sample_volume_hydrostatic_scale = wp.array(
            sample_volume_hydrostatic_scale,
            dtype=float,
            device=self.device,
        )
        self.sample_triangle_rest_area = wp.array(sample_triangle_rest_area, dtype=float, device=self.device)
        self.sample_mass_equiv = wp.zeros(self.sample_count, dtype=float, device=self.device)
        self.sample_wet_weight_same_side = wp.full((self.sample_count,), 1.0, dtype=float, device=self.device)
        self.sample_wet_weight_opposite_side = wp.zeros(self.sample_count, dtype=float, device=self.device)
        self.sample_wet_weight = wp.full((self.sample_count,), 1.0, dtype=float, device=self.device)
        self.sample_wet_weight_same_side_prev = wp.clone(self.sample_wet_weight_same_side)
        self.sample_wet_weight_opposite_side_prev = wp.clone(self.sample_wet_weight_opposite_side)
        self.triangle_wet_weight_same_side = wp.full(model.tri_count, 1.0, dtype=float, device=self.device)
        self.triangle_wet_weight_opposite_side = wp.zeros(model.tri_count, dtype=float, device=self.device)
        self.triangle_wet_weight_same_side_sum = wp.zeros(model.tri_count, dtype=float, device=self.device)
        self.triangle_wet_weight_opposite_side_sum = wp.zeros(model.tri_count, dtype=float, device=self.device)
        triangle_sample_count_per_triangle = [0 for _ in range(model.tri_count)]
        for triangle_index in sample_triangle:
            triangle_sample_count_per_triangle[int(triangle_index)] += 1
        self.triangle_sample_count_per_triangle = wp.array(
            triangle_sample_count_per_triangle,
            dtype=wp.int32,
            device=self.device,
        )

        self.sample_x_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_v_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.sample_normal_world = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_particle_q_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_particle_q_last = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_x_world = wp.zeros(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_x_prev = wp.zeros(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_x_query = wp.zeros(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_aabb_lower = wp.zeros(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_aabb_upper = wp.zeros(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_contact_proxy_motion_max = wp.zeros(1, dtype=float, device=self.device)
        self.sample_force = wp.zeros(self.sample_count, dtype=wp.vec3, device=self.device)
        self.vertex_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.vertex_pressure_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.vertex_velocity_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.vertex_contact_delta = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        body_count = int(getattr(model, "body_count", 0))
        self.body_force = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_torque = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_force_step_sum = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_torque_step_sum = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_force_step_avg = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_torque_step_avg = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.body_force_step_max_norm = wp.zeros(body_count, dtype=float, device=self.device)
        self.body_torque_step_max_norm = wp.zeros(body_count, dtype=float, device=self.device)
        self.body_force_step_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.vertex_contact_delta_generation = -1

        self.boundary_grid = wp.HashGrid(128, 128, 128, device=self.device) if self.sample_count > 0 else None
        self.triangle_contact_grid = wp.HashGrid(128, 128, 128, device=self.device) if self.triangle_count > 0 else None
        self.triangle_contact_bvh = None

        self._empty_body_q = wp.zeros(0, dtype=wp.transform, device=self.device)
        self._empty_body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=self.device)
        self._empty_body_com = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self._triangle_contact_history_initialized = False

    def initialize_deformable_contact_history(self, state: State) -> None:
        """Initialize deformable swept-contact history from a state."""
        if self.triangle_count == 0 or state.particle_q is None:
            return
        if self._triangle_contact_history_initialized:
            return

        self.triangle_contact_particle_q_prev.assign(state.particle_q)
        self.triangle_contact_particle_q_last.assign(state.particle_q)
        self._triangle_contact_history_initialized = True

    def reset_deformable_contact_history(self) -> None:
        """Clear deformable swept-contact history."""
        if self.triangle_count == 0:
            return

        self.triangle_contact_particle_q_prev.zero_()
        self.triangle_contact_particle_q_last.zero_()
        self.triangle_contact_x_prev.zero_()
        self.triangle_contact_x_query.zero_()
        self.triangle_contact_aabb_lower.zero_()
        self.triangle_contact_aabb_upper.zero_()
        self.triangle_contact_proxy_motion_max.zero_()
        self._triangle_contact_history_initialized = False

    def reset_wetness_state(self) -> None:
        """Reset cached deformable wetness state to the default one-sided shell state."""
        self.sample_wet_weight_same_side.fill_(1.0)
        self.sample_wet_weight_opposite_side.zero_()
        self.sample_wet_weight.fill_(1.0)
        self.sample_wet_weight_same_side_prev.fill_(1.0)
        self.sample_wet_weight_opposite_side_prev.zero_()
        self.triangle_wet_weight_same_side.fill_(1.0)
        self.triangle_wet_weight_opposite_side.zero_()
        self.triangle_wet_weight_same_side_sum.zero_()
        self.triangle_wet_weight_opposite_side_sum.zero_()

    def set_triangle_boundary_samples_active(self, active: bool) -> None:
        """Enable or disable deformable triangle boundary samples in density/pressure paths."""
        if self.triangle_sample_count == 0:
            return

        flags = self.sample_flags.numpy()
        triangle_mask = self.sample_triangle.numpy() >= 0
        if active:
            flags[triangle_mask] |= int(BoundarySampleFlags.ACTIVE)
        else:
            flags[triangle_mask] &= ~int(BoundarySampleFlags.ACTIVE)

        self.sample_flags = wp.array(flags, dtype=wp.int32, device=self.device)

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

        if self.triangle_sample_count > 0:
            if state.particle_q is None or state.particle_qd is None:
                raise ValueError("Triangle boundary samples require particle positions and velocities.")

            if not self._triangle_contact_history_initialized:
                self.initialize_deformable_contact_history(state)
            else:
                self.triangle_contact_particle_q_prev.assign(self.triangle_contact_particle_q_last)
                self.triangle_contact_particle_q_last.assign(state.particle_q)

            wp.launch(
                update_deformable_boundary_sample_world_kinematics,
                dim=self.sample_count,
                inputs=[
                    self.sample_triangle,
                    self.sample_vertex0,
                    self.sample_vertex1,
                    self.sample_vertex2,
                    self.sample_barycentric,
                    self.sample_offset_distance,
                    state.particle_q,
                    state.particle_qd,
                ],
                outputs=[
                    self.sample_x_world,
                    self.sample_v_world,
                    self.sample_normal_world,
                ],
                device=self.device,
            )
            wp.launch(
                update_deformable_boundary_sample_hydrostatic_state,
                dim=self.sample_count,
                inputs=[
                    self.sample_triangle,
                    self.sample_vertex0,
                    self.sample_vertex1,
                    self.sample_vertex2,
                    state.particle_q,
                    self.sample_triangle_rest_area,
                    self.sample_area_patch_rest,
                    self.sample_volume_rest,
                    self.sample_volume_quadrature_rest,
                    self.sample_volume_hydrostatic_rest,
                ],
                outputs=[
                    self.sample_area_patch,
                    self.sample_volume,
                    self.sample_volume_quadrature,
                    self.sample_volume_hydrostatic,
                ],
                device=self.device,
            )

            if self.triangle_count > 0:
                wp.launch(
                    update_deformable_triangle_contact_proxy_world_kinematics,
                    dim=self.triangle_count,
                    inputs=[
                        self.triangle_indices,
                        self.model.tri_indices,
                        self.triangle_contact_particle_q_prev,
                    ],
                    outputs=[
                        self.triangle_contact_x_prev,
                    ],
                    device=self.device,
                )
                wp.launch(
                    update_deformable_triangle_contact_proxy_world_kinematics,
                    dim=self.triangle_count,
                    inputs=[
                        self.triangle_indices,
                        self.model.tri_indices,
                        state.particle_q,
                    ],
                    outputs=[
                        self.triangle_contact_x_world,
                    ],
                    device=self.device,
                )
                self.triangle_contact_proxy_motion_max.zero_()
                wp.launch(
                    update_deformable_triangle_contact_proxy_query_kinematics,
                    dim=self.triangle_count,
                    inputs=[
                        self.triangle_contact_x_prev,
                        self.triangle_contact_x_world,
                    ],
                    outputs=[
                        self.triangle_contact_x_query,
                        self.triangle_contact_proxy_motion_max,
                    ],
                    device=self.device,
                )
                wp.launch(
                    update_deformable_triangle_contact_swept_aabbs,
                    dim=self.triangle_count,
                    inputs=[
                        self.triangle_indices,
                        self.model.tri_indices,
                        self.triangle_contact_particle_q_prev,
                        state.particle_q,
                        self.triangle_contact_aabb_padding,
                    ],
                    outputs=[
                        self.triangle_contact_aabb_lower,
                        self.triangle_contact_aabb_upper,
                    ],
                    device=self.device,
                )

    def build_grid(self) -> None:
        """Build boundary neighbor-search structures for sample and cloth contact queries."""
        with wp.ScopedDevice(self.device):
            if self.boundary_grid is not None and self.sample_count > 0:
                self.boundary_grid.build(self.sample_x_world, radius=self.support_radius)
            if self.triangle_contact_grid is not None and self.triangle_count > 0:
                self.triangle_contact_grid.build(self.triangle_contact_x_query, radius=self.support_radius)
            if self.triangle_count > 0:
                if self.triangle_contact_bvh is None:
                    self.triangle_contact_bvh = wp.Bvh(
                        self.triangle_contact_aabb_lower,
                        self.triangle_contact_aabb_upper,
                    )
                else:
                    self.triangle_contact_bvh.refit()

    def clear_forces(self) -> None:
        """Clear boundary sample, body wrench, and deformable reaction accumulators."""
        self.vertex_contact_delta_generation += 1
        self.sample_force.zero_()
        self.vertex_force.zero_()
        self.vertex_pressure_force.zero_()
        self.vertex_velocity_force.zero_()
        self.vertex_contact_delta.zero_()
        self.body_force.zero_()
        self.body_torque.zero_()

    def clear_step_diagnostics(self) -> None:
        """Clear step-level FSI reaction diagnostics."""
        self.body_force_step_sum.zero_()
        self.body_torque_step_sum.zero_()
        self.body_force_step_avg.zero_()
        self.body_torque_step_avg.zero_()
        self.body_force_step_max_norm.zero_()
        self.body_torque_step_max_norm.zero_()
        self.body_force_step_count.zero_()

    def accumulate_step_diagnostics(self) -> None:
        """Accumulate the current body wrench into step-level diagnostics."""
        if self.body_force.shape[0] == 0:
            return

        wp.launch(
            accumulate_step_reaction_diagnostics,
            dim=self.body_force.shape[0],
            inputs=[
                self.body_force,
                self.body_torque,
            ],
            outputs=[
                self.body_force_step_sum,
                self.body_torque_step_sum,
                self.body_force_step_max_norm,
                self.body_torque_step_max_norm,
                self.body_force_step_count,
            ],
            device=self.device,
        )
        wp.launch(
            update_step_reaction_averages,
            dim=self.body_force.shape[0],
            inputs=[
                self.body_force_step_sum,
                self.body_torque_step_sum,
                self.body_force_step_count,
            ],
            outputs=[
                self.body_force_step_avg,
                self.body_torque_step_avg,
            ],
            device=self.device,
        )

    @classmethod
    def _sample_model_shapes(
        cls,
        model: Model,
        spacing: float,
        *,
        shape_indices: Sequence[int] | None,
        include_static: bool,
        include_dynamic: bool,
    ) -> tuple[
        list[int],
        list[int],
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
        list[float],
        list[float],
    ]:
        if model.shape_count == 0:
            return [], [], [], [], [], []

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
        sample_area_patch: list[float] = []
        sample_volume_quadrature: list[float] = []

        for shape_idx in indices:
            if shape_idx < 0 or shape_idx >= model.shape_count:
                raise ValueError(f"Shape index {shape_idx} is out of bounds for {model.shape_count} shapes.")

            geo_type = int(shape_type[shape_idx])
            if geo_type not in (int(GeoType.BOX), int(GeoType.SPHERE)):
                continue

            body = int(shape_body[shape_idx])
            if body < 0 and not include_static:
                continue
            if body >= 0 and not include_dynamic:
                continue

            X_parent_shape = wp.transform(*shape_transform[shape_idx])
            if geo_type == int(GeoType.BOX):
                hx = float(shape_scale[shape_idx][0])
                hy = float(shape_scale[shape_idx][1])
                hz = float(shape_scale[shape_idx][2])
                xs = cls._axis_samples(hx, spacing)
                ys = cls._axis_samples(hy, spacing)
                zs = cls._axis_samples(hz, spacing)
                x_widths = cls._axis_patch_widths(xs)
                y_widths = cls._axis_patch_widths(ys)
                z_widths = cls._axis_patch_widths(zs)
                shape_samples = cls._sample_box_surface(
                    hx,
                    hy,
                    hz,
                    spacing,
                    xs=xs,
                    ys=ys,
                    zs=zs,
                    x_widths=x_widths,
                    y_widths=y_widths,
                    z_widths=z_widths,
                )
            else:
                shape_samples = cls._sample_sphere_surface(float(shape_scale[shape_idx][0]), spacing)

            for point_shape, normal_shape, patch_area, quadrature_volume in shape_samples:
                point_parent = wp.transform_point(X_parent_shape, wp.vec3(*point_shape))
                normal_parent = wp.normalize(wp.transform_vector(X_parent_shape, wp.vec3(*normal_shape)))

                sample_body.append(body)
                sample_shape.append(int(shape_idx))
                sample_x_local.append(cls._vec3_to_tuple(point_parent))
                sample_normal_local.append(cls._vec3_to_tuple(normal_parent))
                sample_area_patch.append(float(patch_area))
                sample_volume_quadrature.append(float(quadrature_volume))

        return (
            sample_body,
            sample_shape,
            sample_x_local,
            sample_normal_local,
            sample_area_patch,
            sample_volume_quadrature,
        )

    @classmethod
    def _sample_model_triangles(
        cls,
        model: Model,
        spacing: float,
        *,
        thickness: float,
        triangle_indices: Sequence[int] | None,
        include_triangles: bool,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        list[tuple[float, float, float]],
        list[int],
        list[int],
        list[tuple[float, float, float]],
        list[tuple[float, float, float]],
        list[float],
        list[float],
        list[float],
        list[float],
    ]:
        if not include_triangles or model.tri_count == 0:
            return [], [], [], [], [], [], [], [], [], [], [], [], []
        if model.tri_indices is None or model.particle_q is None:
            return [], [], [], [], [], [], [], [], [], [], [], [], []

        tri_indices = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        particle_q = np.asarray(model.particle_q.numpy(), dtype=np.float32)

        if triangle_indices is None:
            indices = range(model.tri_count)
        else:
            indices = triangle_indices

        sample_triangle: list[int] = []
        sample_vertex0: list[int] = []
        sample_vertex1: list[int] = []
        sample_vertex2: list[int] = []
        sample_barycentric: list[tuple[float, float, float]] = []
        sample_body: list[int] = []
        sample_shape: list[int] = []
        sample_x_local: list[tuple[float, float, float]] = []
        sample_normal_local: list[tuple[float, float, float]] = []
        sample_offset_distance: list[float] = []
        sample_area_patch: list[float] = []
        sample_volume_quadrature: list[float] = []
        sample_rest_area: list[float] = []

        for triangle_index in indices:
            if triangle_index < 0 or triangle_index >= model.tri_count:
                raise ValueError(f"Triangle index {triangle_index} is out of bounds for {model.tri_count} triangles.")

            vertices = tri_indices[int(triangle_index)]
            p0 = particle_q[int(vertices[0])]
            p1 = particle_q[int(vertices[1])]
            p2 = particle_q[int(vertices[2])]
            normal = np.cross(p1 - p0, p2 - p0)
            normal_norm = float(np.linalg.norm(normal))
            area = 0.5 * normal_norm
            if area <= 0.0:
                continue

            normal = normal / normal_norm
            barycentric_samples = cls._triangle_barycentric_samples(area, spacing)
            patch_area = area / len(barycentric_samples)
            sample_volume = 0.5 * patch_area * thickness
            shell_offset = 0.5 * thickness * normal

            for barycentric in barycentric_samples:
                bary = np.asarray(barycentric, dtype=np.float32)
                mid_point = bary[0] * p0 + bary[1] * p1 + bary[2] * p2

                # Represent deformable cloth as a thin shell with two offset
                # surfaces instead of a single mid-surface point sample.
                # This gives IPBF a wetted-side boundary sample before the
                # cloth mid-surface has already deeply penetrated the fluid.
                for side in (-1.0, 1.0):
                    point = mid_point + side * shell_offset
                    normal_side = side * normal

                    sample_triangle.append(int(triangle_index))
                    sample_vertex0.append(int(vertices[0]))
                    sample_vertex1.append(int(vertices[1]))
                    sample_vertex2.append(int(vertices[2]))
                    sample_barycentric.append(tuple(float(value) for value in barycentric))
                    sample_body.append(-1)
                    sample_shape.append(-1)
                    sample_x_local.append(tuple(float(value) for value in point))
                    sample_normal_local.append(tuple(float(value) for value in normal_side))
                    sample_offset_distance.append(float(side * 0.5 * thickness))
                    sample_area_patch.append(float(patch_area))
                    sample_volume_quadrature.append(float(sample_volume))
                    sample_rest_area.append(float(area))

        return (
            sample_triangle,
            sample_vertex0,
            sample_vertex1,
            sample_vertex2,
            sample_barycentric,
            sample_body,
            sample_shape,
            sample_x_local,
            sample_normal_local,
            sample_offset_distance,
            sample_area_patch,
            sample_volume_quadrature,
            sample_rest_area,
        )

    @staticmethod
    def _compute_triangle_contact_radii(model: Model, triangle_indices: Sequence[int]) -> list[float]:
        """Return conservative centroid radii for sampled deformable triangles."""
        if len(triangle_indices) == 0 or model.tri_indices is None or model.particle_q is None:
            return []

        tri_indices = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        particle_q = np.asarray(model.particle_q.numpy(), dtype=np.float32)

        radii: list[float] = []
        for triangle_index in triangle_indices:
            vertices = tri_indices[int(triangle_index)]
            points = particle_q[vertices]
            centroid = np.mean(points, axis=0)
            radii.append(float(np.linalg.norm(points - centroid, axis=1).max()))

        return radii

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
        sample_area_patch: Sequence[float],
        sample_volume_quadrature: Sequence[float],
        hydrostatic_volume_mode: int,
    ) -> tuple[list[float], list[float]]:
        if len(sample_volume) == 0:
            return [], []

        raw_volume = np.asarray(sample_volume, dtype=np.float32)
        area_patch = np.asarray(sample_area_patch, dtype=np.float32)
        volume_quadrature = np.asarray(sample_volume_quadrature, dtype=np.float32)
        volume_hydrostatic = raw_volume.copy()
        hydrostatic_scale = np.ones(len(sample_volume), dtype=np.float32)

        if hydrostatic_volume_mode == int(cls.HydrostaticVolumeMode.NONE):
            return volume_hydrostatic.tolist(), hydrostatic_scale.tolist()

        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        body_flags = model.body_flags.numpy() if model.body_flags is not None else None

        sample_indices_by_shape: dict[int, list[int]] = {}
        for sample_index, shape_index in enumerate(sample_shape):
            if shape_index < 0:
                continue
            sample_indices_by_shape.setdefault(int(shape_index), []).append(sample_index)

        for shape_index, indices in sample_indices_by_shape.items():
            geo_type = int(shape_type[shape_index])
            if geo_type not in (int(GeoType.BOX), int(GeoType.SPHERE)):
                continue

            body = int(shape_body[shape_index])
            if body < 0:
                continue

            if hydrostatic_volume_mode in (
                int(cls.HydrostaticVolumeMode.DYNAMIC_SHAPE_VOLUME),
                int(cls.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE),
                int(cls.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_THICKNESS),
                int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_VOLUME),
                int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_QUADRATURE),
                int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_THICKNESS),
            ):
                if body_flags is None or (int(body_flags[body]) & int(BodyFlags.KINEMATIC)) != 0:
                    continue

            if hydrostatic_volume_mode in (
                int(cls.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE),
                int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_QUADRATURE),
            ):
                target_volume = volume_quadrature[indices]
                volume_hydrostatic[indices] = target_volume
                hydrostatic_scale[indices] = np.divide(
                    target_volume,
                    raw_volume[indices],
                    out=np.ones_like(target_volume),
                    where=raw_volume[indices] > 0.0,
                )
                continue

            if hydrostatic_volume_mode in (
                int(cls.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_THICKNESS),
                int(cls.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_THICKNESS),
            ):
                patch_area = area_patch[indices]
                patch_area_sum = float(np.sum(patch_area))
                raw_sum = float(np.sum(raw_volume[indices]))
                if patch_area_sum <= 0.0 or raw_sum <= 0.0:
                    continue

                effective_thickness = raw_sum / patch_area_sum
                target_volume = patch_area * effective_thickness
                volume_hydrostatic[indices] = target_volume
                hydrostatic_scale[indices] = np.divide(
                    target_volume,
                    raw_volume[indices],
                    out=np.ones_like(target_volume),
                    where=raw_volume[indices] > 0.0,
                )
                continue

            target_volume = cls._shape_geometric_volume(geo_type, shape_scale[shape_index])
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
    def _shape_geometric_volume(geo_type: int, scale: wp.vec3) -> float:
        if geo_type == int(GeoType.BOX):
            return 8.0 * float(scale[0]) * float(scale[1]) * float(scale[2])
        if geo_type == int(GeoType.SPHERE):
            radius = float(scale[0])
            return (4.0 / 3.0) * math.pi * radius * radius * radius
        return 0.0

    @staticmethod
    def _sample_sphere_surface(
        radius: float,
        spacing: float,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float, float]]:
        if radius <= 0.0:
            return []

        sample_count = max(12, int(math.ceil((4.0 * math.pi * radius * radius) / (spacing * spacing))))
        patch_area = 4.0 * math.pi * radius * radius / sample_count
        quadrature_volume = patch_area * radius / 3.0
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))

        samples: list[tuple[tuple[float, float, float], tuple[float, float, float], float, float]] = []
        for sample_index in range(sample_count):
            y = 1.0 - 2.0 * (sample_index + 0.5) / sample_count
            r_xy = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * sample_index
            normal = (
                math.cos(theta) * r_xy,
                y,
                math.sin(theta) * r_xy,
            )
            point = tuple(radius * component for component in normal)
            samples.append(
                (
                    tuple(float(component) for component in point),
                    tuple(float(component) for component in normal),
                    float(patch_area),
                    float(quadrature_volume),
                )
            )

        return samples

    @staticmethod
    def _triangle_barycentric_samples(area: float, spacing: float) -> list[tuple[float, float, float]]:
        if area <= 0.0:
            return []

        sample_count = max(1, int(math.ceil(area / (spacing * spacing))))
        if sample_count == 1:
            return [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)]

        golden_ratio_fraction = (math.sqrt(5.0) - 1.0) * 0.5
        samples: list[tuple[float, float, float]] = []
        for sample_index in range(sample_count):
            u = (sample_index + 0.5) / sample_count
            v = (sample_index * golden_ratio_fraction) % 1.0
            sqrt_u = math.sqrt(u)
            b0 = 1.0 - sqrt_u
            b1 = sqrt_u * (1.0 - v)
            b2 = sqrt_u * v
            samples.append((float(b0), float(b1), float(b2)))

        return samples

    @staticmethod
    def _sample_box_surface(
        hx: float,
        hy: float,
        hz: float,
        spacing: float,
        *,
        xs: np.ndarray | None = None,
        ys: np.ndarray | None = None,
        zs: np.ndarray | None = None,
        x_widths: np.ndarray | None = None,
        y_widths: np.ndarray | None = None,
        z_widths: np.ndarray | None = None,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float, float]]:
        xs = FSIBoundaryModel._axis_samples(hx, spacing) if xs is None else xs
        ys = FSIBoundaryModel._axis_samples(hy, spacing) if ys is None else ys
        zs = FSIBoundaryModel._axis_samples(hz, spacing) if zs is None else zs
        x_widths = FSIBoundaryModel._axis_patch_widths(xs) if x_widths is None else x_widths
        y_widths = FSIBoundaryModel._axis_patch_widths(ys) if y_widths is None else y_widths
        z_widths = FSIBoundaryModel._axis_patch_widths(zs) if z_widths is None else z_widths

        samples: list[tuple[tuple[float, float, float], tuple[float, float, float], float, float]] = []
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
                    patch_area = FSIBoundaryModel._box_surface_patch_area(
                        ix,
                        iy,
                        iz,
                        len(xs),
                        len(ys),
                        len(zs),
                        x_widths,
                        y_widths,
                        z_widths,
                    )
                    quadrature_volume = FSIBoundaryModel._box_surface_quadrature_volume(
                        ix,
                        iy,
                        iz,
                        len(xs),
                        len(ys),
                        len(zs),
                        hx,
                        hy,
                        hz,
                        x_widths,
                        y_widths,
                        z_widths,
                    )
                    samples.append(
                        (
                            (float(x), float(y), float(z)),
                            tuple(float(v) for v in normal),
                            patch_area,
                            quadrature_volume,
                        )
                    )

        return samples

    @staticmethod
    def _axis_samples(half_extent: float, spacing: float) -> np.ndarray:
        if half_extent <= 0.0:
            return np.array([0.0], dtype=np.float32)

        segments = max(1, int(math.ceil((2.0 * half_extent) / spacing)))
        return np.linspace(-half_extent, half_extent, segments + 1, dtype=np.float32)

    @staticmethod
    def _axis_patch_widths(axis_samples: np.ndarray) -> np.ndarray:
        count = len(axis_samples)
        if count <= 1:
            return np.zeros(count, dtype=np.float32)

        widths = np.empty(count, dtype=np.float32)
        widths[0] = 0.5 * float(axis_samples[1] - axis_samples[0])
        widths[-1] = 0.5 * float(axis_samples[-1] - axis_samples[-2])
        if count > 2:
            widths[1:-1] = 0.5 * (axis_samples[2:] - axis_samples[:-2])
        return widths

    @staticmethod
    def _box_surface_quadrature_volume(
        ix: int,
        iy: int,
        iz: int,
        x_count: int,
        y_count: int,
        z_count: int,
        hx: float,
        hy: float,
        hz: float,
        x_widths: np.ndarray,
        y_widths: np.ndarray,
        z_widths: np.ndarray,
    ) -> float:
        volume = 0.0
        if FSIBoundaryModel._surface_axis_normal(ix, x_count) != 0.0:
            volume += (hx / 3.0) * float(y_widths[iy]) * float(z_widths[iz])
        if FSIBoundaryModel._surface_axis_normal(iy, y_count) != 0.0:
            volume += (hy / 3.0) * float(x_widths[ix]) * float(z_widths[iz])
        if FSIBoundaryModel._surface_axis_normal(iz, z_count) != 0.0:
            volume += (hz / 3.0) * float(x_widths[ix]) * float(y_widths[iy])
        return volume

    @staticmethod
    def _box_surface_patch_area(
        ix: int,
        iy: int,
        iz: int,
        x_count: int,
        y_count: int,
        z_count: int,
        x_widths: np.ndarray,
        y_widths: np.ndarray,
        z_widths: np.ndarray,
    ) -> float:
        area = 0.0
        if FSIBoundaryModel._surface_axis_normal(ix, x_count) != 0.0:
            area += float(y_widths[iy]) * float(z_widths[iz])
        if FSIBoundaryModel._surface_axis_normal(iy, y_count) != 0.0:
            area += float(x_widths[ix]) * float(z_widths[iz])
        if FSIBoundaryModel._surface_axis_normal(iz, z_count) != 0.0:
            area += float(x_widths[ix]) * float(y_widths[iy])
        return area

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
