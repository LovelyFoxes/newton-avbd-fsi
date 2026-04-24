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

"""Position-Based Fluids solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..ipbf.ipbf_kernels import (
    accumulate_particle_shape_boundary_velocity_projection,
    accumulate_particle_shape_boundary_velocity_projection_with_reaction,
    apply_relaxed_jacobi_update,
    apply_viscosity_velocity_diffusion,
    apply_viscosity_velocity_diffusion_with_boundary,
    apply_viscosity_velocity_diffusion_without_grid,
    apply_viscosity_velocity_diffusion_without_grid_with_boundary,
    apply_xsph_velocity_smoothing,
    apply_xsph_velocity_smoothing_with_boundary,
    apply_xsph_velocity_smoothing_without_grid,
    compute_density_and_neighbor_count,
    compute_density_and_neighbor_count_with_boundary,
    finalize_particle_shape_boundary_velocity_projection,
    initialize_density_and_neighbor_count,
    initialize_density_and_neighbor_count_with_boundary,
    initialize_guess_positions,
    initialize_particle_shape_boundary_velocity_projection,
    predict_inertial_positions,
    project_particle_shape_contacts,
    project_particle_shape_contacts_with_reaction,
    restore_non_fluid_particle_state,
    update_velocity_from_positions,
)
from ..solver import SolverBase
from .pbf_kernels import (
    KERNEL_FAMILY_CUBIC_SPLINE,
    KERNEL_FAMILY_POLY6,
    compute_pbf_delta,
    compute_pbf_delta_with_boundary,
    compute_pbf_lambda,
    compute_pbf_lambda_with_boundary,
    initialize_constraint_and_lambda,
    mask_pbf_particle_flags,
)

if TYPE_CHECKING:
    from ..fsi import FSIBoundaryModel

__all__ = ["SolverPBF"]


class SolverPBF(SolverBase):
    """Classic position-based fluids solver.

    This solver implements a standard PBF density-constraint iteration while
    reusing the existing Newton rigid contact, boundary-sample, viscosity, and
    XSPH post-processing paths so comparisons against :class:`SolverIPBF`
    differ mainly in the fluid pressure solve itself.

    Args:
        model: The model to solve.
        config: Optional solver configuration.
        boundary_model: Optional rigid FSI boundary-sample model used for
            boundary-density support and equal-opposite rigid reaction forces.
    """

    @dataclass
    class Config:
        """PBF solver configuration."""

        class KernelFamily(IntEnum):
            """Selectable SPH kernel families for PBF."""

            CUBIC_SPLINE = KERNEL_FAMILY_CUBIC_SPLINE
            POLY6 = KERNEL_FAMILY_POLY6

        rest_density: float = 1000.0
        smoothing_radius: float = 0.1
        kernel_family: KernelFamily = KernelFamily.CUBIC_SPLINE
        iterations: int = 4
        relaxation: float = 1.0
        lambda_regularization: float = 1.0e-6
        use_constraint_clamp: bool = True
        tensile_instability_scale: float = 0.0
        tensile_instability_distance_ratio: float = 0.3
        tensile_instability_power: float = 4.0
        viscosity_coefficient: float = 0.0
        viscosity_boundary_coefficient: float = 0.0
        xsph_coefficient: float = 0.0
        xsph_boundary_coefficient: float = 0.0
        boundary_velocity_damping: float = 1.0
        fsi_reaction_relaxation: float | None = None
        fsi_projection_reaction_relaxation: float | None = None
        fsi_velocity_projection_reaction_relaxation: float | None = None
        fsi_pressure_reaction_relaxation: float = 1.0
        fsi_static_boundary_weight: float = 1.0
        fluid_particle_start: int = 0
        fluid_particle_count: int | None = None

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register PBF-specific custom attributes in the ``pbf`` namespace."""
        model_attributes = [
            ModelBuilder.CustomAttribute(
                name="rest_density",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1000.0,
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="smoothing_radius",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.1,
                namespace="pbf",
            ),
        ]

        state_attributes = [
            ModelBuilder.CustomAttribute(
                name="y",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="x_guess",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="x_new",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="density",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="neighbor_count",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.int32,
                default=0,
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="constraint",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="lambda_value",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="pbf",
            ),
            ModelBuilder.CustomAttribute(
                name="delta_q",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="pbf",
            ),
        ]

        for attribute in model_attributes + state_attributes:
            builder.add_custom_attribute(attribute)

    def __init__(
        self,
        model: Model,
        config: Config | None = None,
        boundary_model: FSIBoundaryModel | None = None,
    ):
        super().__init__(model)

        self.config = config if config is not None else self.Config()
        self.rest_density = float(self.config.rest_density)
        self.smoothing_radius = float(self.config.smoothing_radius)
        self.kernel_family = int(self.Config.KernelFamily(self.config.kernel_family))
        self.iterations = int(self.config.iterations)
        self.relaxation = float(self.config.relaxation)
        self.lambda_regularization = float(self.config.lambda_regularization)
        self.use_constraint_clamp = bool(self.config.use_constraint_clamp)
        self.tensile_instability_scale = float(self.config.tensile_instability_scale)
        self.tensile_instability_distance_ratio = float(self.config.tensile_instability_distance_ratio)
        self.tensile_instability_power = float(self.config.tensile_instability_power)
        self.viscosity_coefficient = float(self.config.viscosity_coefficient)
        self.viscosity_boundary_coefficient = float(self.config.viscosity_boundary_coefficient)
        self.xsph_coefficient = float(self.config.xsph_coefficient)
        self.xsph_boundary_coefficient = float(self.config.xsph_boundary_coefficient)
        self.boundary_velocity_damping = float(self.config.boundary_velocity_damping)
        legacy_reaction_relaxation = self.config.fsi_reaction_relaxation
        projection_reaction_relaxation = self.config.fsi_projection_reaction_relaxation
        velocity_projection_reaction_relaxation = self.config.fsi_velocity_projection_reaction_relaxation
        self.fsi_projection_reaction_relaxation = float(
            projection_reaction_relaxation
            if projection_reaction_relaxation is not None
            else (legacy_reaction_relaxation if legacy_reaction_relaxation is not None else 1.0)
        )
        self.fsi_velocity_projection_reaction_relaxation = float(
            velocity_projection_reaction_relaxation
            if velocity_projection_reaction_relaxation is not None
            else (legacy_reaction_relaxation if legacy_reaction_relaxation is not None else 1.0)
        )
        self.fsi_pressure_reaction_relaxation = float(self.config.fsi_pressure_reaction_relaxation)
        self.fsi_static_boundary_weight = float(self.config.fsi_static_boundary_weight)
        self.fluid_particle_start, self.fluid_particle_count = self._validate_fluid_particle_range()
        self._uses_particle_subset = (
            self.fluid_particle_start != 0 or self.fluid_particle_count != self.model.particle_count
        )
        self._max_particle_radius = self._compute_max_particle_radius()
        self.boundary_model = None
        self.set_boundary_model(boundary_model)

        if not hasattr(model, "pbf"):
            raise ValueError(
                "SolverPBF requires PBF custom attributes. "
                "Call SolverPBF.register_custom_attributes(builder) before finalize()."
            )

        model.pbf.rest_density.fill_(self.rest_density)
        model.pbf.smoothing_radius.fill_(self.smoothing_radius)

        self._initial_state = model.state()
        with wp.ScopedDevice(model.device):
            self._empty_body_q = wp.empty(0, dtype=wp.transform)
            self._empty_body_com = wp.empty(0, dtype=wp.vec3)
            self._boundary_projected_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projected_contact_count = wp.zeros(model.particle_count, dtype=wp.int32)
            self._viscosity_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._xsph_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_density = wp.zeros(model.particle_count, dtype=float)
            self._boundary_neighbor_count = wp.zeros(model.particle_count, dtype=wp.int32)
            self._boundary_projection_delta = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projection_delta_total = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._pbf_particle_flags = wp.empty(model.particle_count, dtype=wp.int32)
            self._refresh_particle_flags()

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    def set_boundary_model(self, boundary_model: FSIBoundaryModel | None) -> None:
        """Set the optional rigid boundary-sample model used by PBF."""
        if boundary_model is not None:
            if getattr(boundary_model, "model", self.model) is not self.model:
                raise ValueError("PBF boundary model must be built from this solver's model.")
            if getattr(boundary_model, "device", self.model.device) != self.model.device:
                raise ValueError("PBF boundary model device must match the solver model device.")

        self.boundary_model = boundary_model

    def _validate_fluid_particle_range(self) -> tuple[int, int]:
        """Return the validated consecutive particle range solved by PBF."""
        model_particle_count = int(self.model.particle_count)
        start = int(self.config.fluid_particle_start)
        if start < 0 or start > model_particle_count:
            raise ValueError("PBF fluid particle start must be within the model particle range.")

        configured_count = self.config.fluid_particle_count
        if configured_count is None:
            count = model_particle_count - start
        else:
            count = int(configured_count)

        if count < 0 or start + count > model_particle_count:
            raise ValueError("PBF fluid particle count must fit inside the model particle range.")

        return start, count

    def _compute_max_particle_radius(self) -> float:
        """Return the largest model particle radius [m]."""
        if self.model.particle_count == 0 or self.model.particle_radius is None:
            return 0.0

        return float(np.max(self.model.particle_radius.numpy()))

    def _refresh_particle_flags(self) -> None:
        """Refresh solver-local flags that mark only the active PBF fluid range."""
        if self.model.particle_count == 0:
            return

        wp.launch(
            mask_pbf_particle_flags,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_flags,
                self.fluid_particle_start,
                self.fluid_particle_count,
            ],
            outputs=[self._pbf_particle_flags],
            device=self.model.device,
        )

    def _restore_non_fluid_particle_state(self, state_in: State, state_out: State) -> None:
        """Keep particles outside the active PBF fluid range untouched by the solver."""
        if not self._uses_particle_subset or self.model.particle_count == 0:
            return

        wp.launch(
            restore_non_fluid_particle_state,
            dim=self.model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.fluid_particle_start,
                self.fluid_particle_count,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=self.model.device,
        )

    def reset(self, state_out: State) -> None:
        """Reset a state to the solver's initial particle configuration."""
        if state_out.particle_q is None or state_out.particle_qd is None or state_out.particle_f is None:
            raise ValueError("SolverPBF.reset() requires a writable particle state.")

        if not hasattr(state_out, "pbf"):
            raise ValueError(
                "State is missing PBF attributes. Rebuild the model with SolverPBF.register_custom_attributes()."
            )

        state_out.assign(self._initial_state)
        state_out.pbf.y.assign(state_out.particle_q)
        wp.launch(
            initialize_guess_positions,
            dim=self.model.particle_count,
            inputs=[state_out.pbf.y],
            outputs=[state_out.pbf.x_guess, state_out.pbf.x_new],
            device=self.model.device,
        )

        state_out.pbf.density.zero_()
        state_out.pbf.neighbor_count.zero_()
        state_out.pbf.constraint.zero_()
        state_out.pbf.lambda_value.zero_()
        state_out.pbf.delta_q.zero_()
        self._boundary_density.zero_()
        self._boundary_neighbor_count.zero_()
        self._boundary_projection_delta.zero_()
        self._boundary_projection_delta_total.zero_()

        if self.boundary_model is not None:
            self.boundary_model.clear_forces()

        if self.model.particle_count > 1 and self.model.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.model.particle_grid.build(state_out.pbf.x_guess, radius=self.smoothing_radius)

    def _has_shape_boundary_contacts(self, contacts: Contacts | None) -> bool:
        """Return whether shape boundary projection can be applied."""
        return contacts is not None and self.model.shape_count > 0 and contacts.soft_contact_max > 0

    def _has_fsi_body_reaction_target(self, state: State) -> bool:
        """Return whether rigid FSI body reaction accumulation is available."""
        boundary_model = self.boundary_model
        return (
            boundary_model is not None
            and self.model.body_count > 0
            and state.body_q is not None
            and self.model.body_com is not None
            and getattr(boundary_model, "shape_sample_count", None) is not None
        )

    def _begin_step(self, state_in: State, state_out: State, dt: float) -> bool:
        """Prepare PBF buffers for one timestep."""
        if state_in.particle_q is None or state_in.particle_qd is None:
            raise ValueError("SolverPBF requires particle positions and velocities.")

        if state_in.particle_f is None or state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("SolverPBF requires particle forces and writable output particle state.")

        if not hasattr(state_out, "pbf"):
            raise ValueError(
                "State is missing PBF attributes. Rebuild the model with SolverPBF.register_custom_attributes()."
            )

        model = self.model
        if model.particle_count == 0 or self.fluid_particle_count == 0:
            return False

        self._refresh_particle_flags()
        if self.boundary_model is not None:
            self.boundary_model.clear_forces()
            self._boundary_projection_delta_total.zero_()

        state_out.particle_q.assign(state_in.particle_q)
        state_out.particle_qd.assign(state_in.particle_qd)
        if model.body_count and state_in.body_q is not None and state_out.body_q is not None:
            state_out.body_q.assign(state_in.body_q)
            state_out.body_qd.assign(state_in.body_qd)

        wp.launch(
            predict_inertial_positions,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                self._pbf_particle_flags,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[state_out.pbf.y],
            device=model.device,
        )
        wp.launch(
            initialize_guess_positions,
            dim=model.particle_count,
            inputs=[state_out.pbf.y],
            outputs=[state_out.pbf.x_guess, state_out.pbf.x_new],
            device=model.device,
        )
        return True

    def _solve_zero_iteration(self, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """Apply projection-only work for zero-iteration configurations."""
        self._apply_shape_boundary_contacts(state_out, state_out.pbf.x_guess, contacts, dt)
        state_out.pbf.x_new.assign(state_out.pbf.x_guess)
        self._compute_density_and_lambda(state_out, state_out.pbf.x_guess)
        state_out.pbf.delta_q.zero_()

    def _solve_iteration(
        self,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        iteration: int,
        iteration_count: int | None = None,
    ) -> None:
        """Run one PBF density-constraint iteration."""
        del iteration, iteration_count
        model = self.model

        self._compute_density_and_lambda(state_out, state_out.pbf.x_guess)
        self._compute_position_deltas(state_out, state_out.pbf.x_guess, dt)

        wp.launch(
            apply_relaxed_jacobi_update,
            dim=model.particle_count,
            inputs=[
                self._pbf_particle_flags,
                state_out.pbf.x_guess,
                state_out.pbf.delta_q,
                self.relaxation,
            ],
            outputs=[state_out.pbf.x_new],
            device=model.device,
        )
        self._apply_shape_boundary_contacts(state_out, state_out.pbf.x_new, contacts, dt)
        state_out.pbf.x_guess.assign(state_out.pbf.x_new)

    def _finalize_step(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Commit converged PBF positions and reconstruct particle velocities."""
        self._compute_density_and_lambda(state_out, state_out.pbf.x_guess)
        state_out.particle_q.assign(state_out.pbf.x_guess)

        wp.launch(
            update_velocity_from_positions,
            dim=self.model.particle_count,
            inputs=[
                state_out.pbf.x_guess,
                state_in.particle_q,
                self._pbf_particle_flags,
                dt,
            ],
            outputs=[state_out.particle_qd],
            device=self.model.device,
        )

        self._apply_shape_boundary_velocity_projection(state_out, contacts, dt)
        self._apply_viscosity_velocity_diffusion(state_out, dt)
        self._apply_xsph_velocity_smoothing(state_out)
        self._restore_non_fluid_particle_state(state_in, state_out)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by one timestep."""
        del control
        if not self._begin_step(state_in, state_out, dt):
            return

        if self.iterations <= 0:
            self._solve_zero_iteration(state_out, contacts, dt)
            self._finalize_step(state_in, state_out, contacts, dt)
            return

        for iteration in range(self.iterations):
            self._solve_iteration(state_out, contacts, dt, iteration, self.iterations)

        self._finalize_step(state_in, state_out, contacts, dt)

    def _compute_density_and_lambda(self, state: State, particle_q: wp.array[wp.vec3]) -> None:
        """Compute density estimates and PBF lambda multipliers on ``particle_q``."""
        model = self.model
        boundary_model = self.boundary_model
        has_boundary = (
            boundary_model is not None
            and getattr(boundary_model, "sample_count", 0) > 0
            and getattr(boundary_model, "boundary_grid", None) is not None
        )

        self._boundary_density.zero_()
        self._boundary_neighbor_count.zero_()

        if has_boundary:
            boundary_model.update_world_kinematics(state)
            boundary_model.build_grid()

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(particle_q, radius=self.smoothing_radius)

            if has_boundary:
                wp.launch(
                    compute_density_and_neighbor_count_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[
                        state.pbf.density,
                        state.pbf.neighbor_count,
                        self._boundary_density,
                        self._boundary_neighbor_count,
                    ],
                    device=model.device,
                )
                wp.launch(
                    compute_pbf_lambda_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        state.pbf.density,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        int(self.use_constraint_clamp),
                        self.lambda_regularization,
                    ],
                    outputs=[state.pbf.constraint, state.pbf.lambda_value],
                    device=model.device,
                )
            else:
                wp.launch(
                    compute_density_and_neighbor_count,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[state.pbf.density, state.pbf.neighbor_count],
                    device=model.device,
                )
                wp.launch(
                    compute_pbf_lambda,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        state.pbf.density,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        int(self.use_constraint_clamp),
                        self.lambda_regularization,
                    ],
                    outputs=[state.pbf.constraint, state.pbf.lambda_value],
                    device=model.device,
                )
        else:
            if has_boundary:
                wp.launch(
                    initialize_density_and_neighbor_count_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[
                        state.pbf.density,
                        state.pbf.neighbor_count,
                        self._boundary_density,
                        self._boundary_neighbor_count,
                    ],
                    device=model.device,
                )
            else:
                wp.launch(
                    initialize_density_and_neighbor_count,
                    dim=model.particle_count,
                    inputs=[
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[state.pbf.density, state.pbf.neighbor_count],
                    device=model.device,
                )

            wp.launch(
                initialize_constraint_and_lambda,
                dim=model.particle_count,
                inputs=[
                    state.pbf.density,
                    self._pbf_particle_flags,
                    self.rest_density,
                    int(self.use_constraint_clamp),
                ],
                outputs=[state.pbf.constraint, state.pbf.lambda_value],
                device=model.device,
            )

    def _compute_position_deltas(
        self,
        state: State,
        particle_q: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Compute one PBF position-correction field on ``particle_q``."""
        model = self.model
        boundary_model = self.boundary_model
        has_boundary = (
            boundary_model is not None
            and getattr(boundary_model, "sample_count", 0) > 0
            and getattr(boundary_model, "boundary_grid", None) is not None
        )
        tensile_distance = self.tensile_instability_distance_ratio * self.smoothing_radius

        if model.particle_count > 1 and model.particle_grid is not None:
            if has_boundary:
                wp.launch(
                    compute_pbf_delta_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        state.pbf.lambda_value,
                        boundary_model.sample_x_world,
                        boundary_model.sample_body,
                        boundary_model.sample_triangle,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        state.body_q if state.body_q is not None else self._empty_body_q,
                        model.body_com if model.body_com is not None else self._empty_body_com,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.fsi_static_boundary_weight,
                        self.tensile_instability_scale,
                        tensile_distance,
                        self.tensile_instability_power,
                        dt,
                        self.relaxation,
                        self.fsi_pressure_reaction_relaxation,
                    ],
                    outputs=[
                        state.pbf.delta_q,
                        boundary_model.sample_force,
                        boundary_model.body_force,
                        boundary_model.body_torque,
                    ],
                    device=model.device,
                )
            else:
                wp.launch(
                    compute_pbf_delta,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        state.pbf.lambda_value,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.tensile_instability_scale,
                        tensile_distance,
                        self.tensile_instability_power,
                    ],
                    outputs=[state.pbf.delta_q],
                    device=model.device,
                )
        else:
            state.pbf.delta_q.zero_()

    def _apply_shape_boundary_contacts(
        self,
        state: State,
        particle_q: wp.array[wp.vec3],
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Project the given particle positions out of rigid boundary contacts."""
        if not self._has_shape_boundary_contacts(contacts):
            return

        model = self.model
        state.particle_q.assign(particle_q)
        model.collide(state, contacts)

        boundary_model = self.boundary_model
        has_reaction_target = self._has_fsi_body_reaction_target(state)

        if has_reaction_target:
            self._boundary_projection_delta.zero_()
            wp.launch(
                project_particle_shape_contacts_with_reaction,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_q,
                    model.particle_mass,
                    model.particle_radius,
                    self._pbf_particle_flags,
                    state.body_q,
                    model.body_com,
                    model.shape_body,
                    boundary_model.shape_sample_count,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    1.0,
                    dt,
                    self.fsi_projection_reaction_relaxation,
                ],
                outputs=[
                    self._boundary_projection_delta,
                    self._boundary_projection_delta_total,
                    boundary_model.body_force,
                    boundary_model.body_torque,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                project_particle_shape_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    self._pbf_particle_flags,
                    state.body_q if state.body_q is not None else self._empty_body_q,
                    model.shape_body,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    1.0,
                ],
                device=model.device,
            )

    def _apply_shape_boundary_velocity_projection(self, state: State, contacts: Contacts | None, dt: float) -> None:
        """Project final particle velocities against active particle-shape contacts."""
        if not self._has_shape_boundary_contacts(contacts):
            return

        model = self.model
        model.collide(state, contacts)
        wp.launch(
            initialize_particle_shape_boundary_velocity_projection,
            dim=model.particle_count,
            inputs=[self._pbf_particle_flags],
            outputs=[self._boundary_projected_particle_qd, self._boundary_projected_contact_count],
            device=model.device,
        )

        boundary_model = self.boundary_model
        has_reaction_target = self._has_fsi_body_reaction_target(state)
        if has_reaction_target:
            wp.launch(
                accumulate_particle_shape_boundary_velocity_projection_with_reaction,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_qd,
                    model.particle_mass,
                    self._pbf_particle_flags,
                    state.body_q,
                    model.body_com,
                    model.shape_body,
                    boundary_model.shape_sample_count,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    self.boundary_velocity_damping,
                    dt,
                    self.fsi_velocity_projection_reaction_relaxation,
                ],
                outputs=[
                    self._boundary_projected_particle_qd,
                    self._boundary_projected_contact_count,
                    boundary_model.body_force,
                    boundary_model.body_torque,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                accumulate_particle_shape_boundary_velocity_projection,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_qd,
                    self._pbf_particle_flags,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    self.boundary_velocity_damping,
                ],
                outputs=[self._boundary_projected_particle_qd, self._boundary_projected_contact_count],
                device=model.device,
            )

        wp.launch(
            finalize_particle_shape_boundary_velocity_projection,
            dim=model.particle_count,
            inputs=[
                state.particle_qd,
                self._pbf_particle_flags,
                self._boundary_projected_particle_qd,
                self._boundary_projected_contact_count,
            ],
            device=model.device,
        )

    def _apply_xsph_velocity_smoothing(self, state: State) -> None:
        """Apply an optional XSPH-style velocity smoothing pass."""
        if (self.xsph_coefficient <= 0.0 and self.xsph_boundary_coefficient <= 0.0) or self.model.particle_count == 0:
            return

        model = self.model
        boundary_model = self.boundary_model
        has_boundary = (
            boundary_model is not None
            and self.xsph_boundary_coefficient > 0.0
            and getattr(boundary_model, "sample_count", 0) > 0
            and getattr(boundary_model, "boundary_grid", None) is not None
        )

        if model.particle_count > 1 and model.particle_grid is not None:
            if has_boundary:
                wp.launch(
                    apply_xsph_velocity_smoothing_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        state.particle_q,
                        state.particle_qd,
                        state.pbf.density,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_v_world,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.xsph_coefficient,
                        self.xsph_boundary_coefficient,
                    ],
                    outputs=[self._xsph_particle_qd],
                    device=model.device,
                )
            else:
                wp.launch(
                    apply_xsph_velocity_smoothing,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        state.particle_q,
                        state.particle_qd,
                        state.pbf.density,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.xsph_coefficient,
                    ],
                    outputs=[self._xsph_particle_qd],
                    device=model.device,
                )
        else:
            wp.launch(
                apply_xsph_velocity_smoothing_without_grid,
                dim=model.particle_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    state.pbf.density,
                    self._pbf_particle_flags,
                    self.xsph_coefficient,
                ],
                outputs=[self._xsph_particle_qd],
                device=model.device,
            )

        state.particle_qd.assign(self._xsph_particle_qd)

    def _apply_viscosity_velocity_diffusion(self, state: State, dt: float) -> None:
        """Apply an optional SPH viscosity diffusion pass to the final velocities."""
        if (
            (self.viscosity_coefficient <= 0.0 and self.viscosity_boundary_coefficient <= 0.0)
            or self.model.particle_count == 0
            or dt <= 0.0
        ):
            return

        model = self.model
        boundary_model = self.boundary_model
        has_boundary = (
            boundary_model is not None
            and self.viscosity_boundary_coefficient > 0.0
            and getattr(boundary_model, "sample_count", 0) > 0
            and getattr(boundary_model, "boundary_grid", None) is not None
        )

        if model.particle_count > 1 and model.particle_grid is not None:
            if has_boundary:
                wp.launch(
                    apply_viscosity_velocity_diffusion_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        state.particle_q,
                        state.particle_qd,
                        state.pbf.density,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_v_world,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.viscosity_coefficient,
                        self.viscosity_boundary_coefficient,
                        dt,
                    ],
                    outputs=[self._viscosity_particle_qd],
                    device=model.device,
                )
            else:
                wp.launch(
                    apply_viscosity_velocity_diffusion,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        state.particle_q,
                        state.particle_qd,
                        state.pbf.density,
                        model.particle_mass,
                        self._pbf_particle_flags,
                        model.particle_world,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.viscosity_coefficient,
                        dt,
                    ],
                    outputs=[self._viscosity_particle_qd],
                    device=model.device,
                )
        else:
            if has_boundary:
                wp.launch(
                    apply_viscosity_velocity_diffusion_without_grid_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.boundary_grid.id,
                        state.particle_q,
                        state.particle_qd,
                        state.pbf.density,
                        self._pbf_particle_flags,
                        boundary_model.sample_x_world,
                        boundary_model.sample_v_world,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.viscosity_boundary_coefficient,
                        dt,
                    ],
                    outputs=[self._viscosity_particle_qd],
                    device=model.device,
                )
            else:
                wp.launch(
                    apply_viscosity_velocity_diffusion_without_grid,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q,
                        state.particle_qd,
                        self._pbf_particle_flags,
                        self.viscosity_coefficient,
                        dt,
                    ],
                    outputs=[self._viscosity_particle_qd],
                    device=model.device,
                )

        state.particle_qd.assign(self._viscosity_particle_qd)
