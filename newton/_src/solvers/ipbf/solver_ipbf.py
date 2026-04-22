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

"""Implicit Position-Based Fluids solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import (
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
)
from ..solver import SolverBase
from .ipbf_kernels import (
    accumulate_boundary_pressure_reaction,
    accumulate_boundary_triangle_hydrostatic_state,
    accumulate_particle_shape_boundary_velocity_projection,
    accumulate_particle_shape_boundary_velocity_projection_with_reaction,
    accumulate_particle_triangle_contact_corrections,
    accumulate_particle_triangle_contact_corrections_from_grid,
    accumulate_particle_triangle_contact_corrections_from_pairs,
    accumulate_particle_triangle_contact_corrections_if_overflow,
    accumulate_triangle_contact_pair_cache_fluid_displacement_max,
    accumulate_triangle_contact_pair_cache_triangle_displacement_max,
    apply_artificial_damping,
    apply_boundary_triangle_hydrostatic_support,
    apply_particle_triangle_contact_deltas,
    apply_relaxed_jacobi_update,
    apply_viscosity_velocity_diffusion,
    apply_viscosity_velocity_diffusion_with_boundary,
    apply_viscosity_velocity_diffusion_without_grid,
    apply_viscosity_velocity_diffusion_without_grid_with_boundary,
    apply_xsph_velocity_smoothing,
    apply_xsph_velocity_smoothing_with_boundary,
    apply_xsph_velocity_smoothing_without_grid,
    clear_triangle_contact_pair_cache_reuse_for_inactive,
    clear_triangle_contact_pair_cache_reuse_for_triangle_count_change,
    clear_triangle_contact_pair_cache_reuse_for_uncached_particles_near_bvh,
    collect_particle_triangle_contact_pairs_from_bvh,
    compute_constraint_and_gradient,
    compute_constraint_and_gradient_with_boundary,
    compute_density_and_neighbor_count,
    compute_density_and_neighbor_count_with_boundary,
    compute_force,
    compute_force_without_grid,
    compute_hessian,
    compute_hessian_without_grid,
    finalize_particle_shape_boundary_velocity_projection,
    finalize_triangle_contact_pair_cache_collection,
    finalize_triangle_contact_pair_cache_reuse,
    initialize_constraint_and_gradient,
    initialize_constraint_and_gradient_with_boundary,
    initialize_density_and_neighbor_count,
    initialize_density_and_neighbor_count_with_boundary,
    initialize_guess_positions,
    initialize_particle_shape_boundary_velocity_projection,
    initialize_triangle_contact_pair_cache_reuse,
    mark_triangle_contact_pair_cached_particles,
    mask_ipbf_particle_flags,
    predict_inertial_positions,
    prepare_triangle_contact_pair_collection,
    project_particle_shape_contacts,
    project_particle_shape_contacts_with_reaction,
    restore_non_fluid_particle_state,
    solve_local_system,
    update_boundary_triangle_wet_weights,
    update_triangle_contact_pair_cache_snapshot_for_sampled_triangles,
    update_triangle_contact_pair_cache_snapshot_if_active,
    update_velocity_from_positions,
)

if TYPE_CHECKING:
    from ..fsi import FSIBoundaryModel

__all__ = ["SolverIPBF"]


class SolverIPBF(SolverBase):
    """Implicit Position-Based Fluids solver.

    This solver implements an Implicit Position-Based Fluids algorithm,
    roughly following [1].

    [1] https://doi.org/10.1145/3757377.3764005

    Args:
        model: The model to solve.
        config: The solver configuration.
        boundary_model: Optional FSI boundary sample model used to add
            solid-boundary density contributions.

    Returns:
        The solver.
    """

    @dataclass
    class Config:
        """Implicit Position-Based Fluids solver configuration.

        Attributes:
            rest_density: Fluid rest density [kg/m^3].
            smoothing_radius: SPH kernel support radius [m].
            kernel_family: SPH kernel family used for density, gradient, and
                Hessian evaluation.
            compliance: Normalized compliance parameter ``alpha = 1 / k``.
            hessian_regularization: Small diagonal regularization added to the
                local Hessian to keep it invertible.
            iterations: Number of relaxed Jacobi iterations.
            relaxation: Position update relaxation factor. The paper uses 0.5.
            use_constraint_clamp: Clamp negative density constraints at free
                surfaces to zero.
            damping_compliance: Compliance used for the alternative position
                solve in artificial damping.
            damping_beta: Distance threshold factor used by the damping model.
            viscosity_coefficient: Kinematic viscosity coefficient [m^2/s]
                applied as a post-solve SPH velocity diffusion step.
            viscosity_boundary_coefficient: Boundary-sample viscosity diffusion
                coefficient applied using boundary sample velocities when a
                boundary model is active.
            xsph_coefficient: XSPH velocity smoothing coefficient applied after
                the position solve.
            xsph_boundary_coefficient: Boundary-sample XSPH smoothing
                coefficient applied using boundary sample velocities when a
                boundary model is active.
            boundary_velocity_damping: Tangential damping multiplier applied to
                particle velocities at final particle-shape contacts.
            fsi_reaction_relaxation: Legacy compatibility multiplier applied to
                both position- and velocity-projection reaction terms when the
                dedicated reaction multipliers are omitted.
            fsi_projection_reaction_relaxation: Unitless multiplier applied when
                converting particle-shape position corrections into FSI body
                reaction forces and torques.
            fsi_velocity_projection_reaction_relaxation: Unitless multiplier
                applied when converting boundary velocity projection deltas into
                FSI body reaction forces and torques.
            fsi_pressure_reaction_relaxation: Unitless multiplier applied when
                converting boundary pressure-gradient position increments into
                FSI body reaction forces and torques.
            fsi_triangle_hydrostatic_support_scale: Unitless multiplier used
                to scale a supplemental non-contact hydrostatic pressure
                support operator for wet-side triangle boundary samples. Zero
                disables this cloth-specific thin-shell support path.
            fsi_triangle_reconstructed_support_depth_scale: Unitless
                multiplier used to reconstruct a one-sided support volume for
                triangle boundary samples from ``sample_area * support_radius``.
                Zero keeps the raw sampled-shell volume.
            fsi_static_boundary_weight: Unitless diagnostic multiplier applied
                to static boundary-sample contributions in the density,
                constraint-gradient, pressure-reaction, and boundary-aware
                velocity-smoothing / viscosity paths.
            fsi_decouple_triangle_boundary_density: Whether triangle-bound
                deformable FSI boundary samples contribute through a decoupled
                support path instead of being fully mixed into the fluid-side
                density constraint.
            fsi_triangle_contact_enabled: Whether active fluid particles are
                projected against triangle-bound deformable FSI boundaries.
            fsi_triangle_contact_use_bvh: Whether deformable triangle contact
                should query a swept triangle BVH before falling back to the
                legacy proxy-grid or brute-force scan paths.
            fsi_triangle_contact_pair_capacity: Maximum number of compact
                cloth contact candidate pairs stored per triangle-contact
                solve when the swept BVH broadphase is enabled. If ``None``,
                the solver derives a capacity from the active fluid count.
            fsi_triangle_contact_pair_cache_skin: Conservative motion skin
                [m] used to decide whether previously collected compact BVH
                contact pairs may be reused. If ``None``, the solver derives
                a small default from the maximum particle radius.
            fsi_triangle_contact_use_grid: Whether deformable triangle contact
                should use the boundary model's triangle proxy grid instead of
                scanning every sampled triangle when BVH broadphase is
                disabled or unavailable.
            fsi_triangle_contact_search_radius: Candidate triangle proxy query
                radius [m] used by the proxy-grid fallback. If ``None``, a
                conservative radius based on the triangle proxy radius,
                particle radius, and contact margin is used.
            fsi_triangle_contact_continuous_enabled: Whether deformable
                triangle contact should use a swept moving-triangle side-change
                test between the previous and current particle/triangle
                positions to reduce thin-shell leakage.
            fsi_triangle_contact_margin: Additional particle-triangle contact
                margin [m] added to the fluid particle radius.
            fsi_triangle_contact_relaxation: Unitless relaxation used when
                applying symmetric particle-triangle contact corrections.
            fluid_particle_start: First particle index solved as IPBF fluid.
            fluid_particle_count: Number of consecutive particles solved as
                IPBF fluid. If ``None``, all particles from
                :attr:`fluid_particle_start` to the end of the model are solved.
        """

        class KernelFamily(IntEnum):
            """Selectable SPH kernel families for IPBF."""

            CUBIC_SPLINE = 0
            POLY6 = 1

        rest_density: float = 1000.0
        smoothing_radius: float = 0.1
        kernel_family: KernelFamily = KernelFamily.CUBIC_SPLINE
        compliance: float = 0.0
        hessian_regularization: float = 1.0e-6
        iterations: int = 3
        relaxation: float = 0.5
        use_constraint_clamp: bool = True
        damping_compliance: float = 1.0 / 1000.0
        damping_beta: float = 60.0
        viscosity_coefficient: float = 0.0
        viscosity_boundary_coefficient: float = 0.0
        xsph_coefficient: float = 0.0
        xsph_boundary_coefficient: float = 0.0
        boundary_velocity_damping: float = 1.0
        fsi_reaction_relaxation: float | None = None
        fsi_projection_reaction_relaxation: float | None = None
        fsi_velocity_projection_reaction_relaxation: float | None = None
        fsi_pressure_reaction_relaxation: float = 1.0
        fsi_triangle_hydrostatic_support_scale: float = 0.0
        fsi_triangle_reconstructed_support_depth_scale: float = 0.0
        fsi_static_boundary_weight: float = 1.0
        fsi_decouple_triangle_boundary_density: bool = True
        fsi_triangle_contact_enabled: bool = True
        fsi_triangle_contact_use_bvh: bool = True
        fsi_triangle_contact_pair_capacity: int | None = None
        fsi_triangle_contact_pair_cache_skin: float | None = None
        fsi_triangle_contact_use_grid: bool = True
        fsi_triangle_contact_search_radius: float | None = None
        fsi_triangle_contact_continuous_enabled: bool = True
        fsi_triangle_contact_margin: float = 0.0
        fsi_triangle_contact_relaxation: float = 1.0
        fluid_particle_start: int = 0
        fluid_particle_count: int | None = None

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register IPBF-specific custom attributes in the ``ipbf`` namespace.

        This method registers global fluid parameters on the model and per-particle
        iterative state variables for the implicit position-based fluids solver.

        Attributes registered on Model (global):
            - ``ipbf:rest_density``: Fluid rest density [kg/m^3]
            - ``ipbf:smoothing_radius``: SPH kernel support radius [m]
            - ``ipbf:compliance``: Normalized compliance parameter ``alpha = 1 / k``

        Attributes registered on State (per-particle):
            - ``ipbf:y``: Inertial target position :math:`y = x^t + h v^t + h^2 a^*`
            - ``ipbf:x_guess``: Current relaxed Jacobi iterate
            - ``ipbf:x_new``: Updated relaxed Jacobi iterate
            - ``ipbf:x_star``: Alternative final-iteration position used by artificial damping
            - ``ipbf:density``: Current SPH density estimate :math:`\\rho_i` [kg/m^3]
            - ``ipbf:neighbor_count``: Number of active neighboring particles inside the support radius, plus
              boundary samples when a boundary model is enabled
            - ``ipbf:constraint``: Density constraint value :math:`C_i = \\rho_i / \\rho_0 - 1`
            - ``ipbf:constraint_gradient``: Local constraint gradient :math:`\\partial C_i / \\partial x_i`
            - ``ipbf:force``: Local Newton-step force term :math:`f_i`
            - ``ipbf:hessian``: Local 3x3 Hessian approximation :math:`H_i`
            - ``ipbf:delta_q``: Local position increment :math:`\\Delta x_i`
        """

        identity = wp.mat33(np.eye(3))

        model_attributes = [
            ModelBuilder.CustomAttribute(
                name="rest_density",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1000.0,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="smoothing_radius",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.1,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="compliance",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="ipbf",
            ),
        ]

        state_attributes = [
            ModelBuilder.CustomAttribute(
                name="y",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="x_guess",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="x_new",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="x_star",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="density",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="neighbor_count",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.int32,
                default=0,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="constraint",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="constraint_gradient",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="force",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="hessian",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=identity,
                namespace="ipbf",
            ),
            ModelBuilder.CustomAttribute(
                name="delta_q",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.STATE,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="ipbf",
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
        self.compliance = float(self.config.compliance)
        self.hessian_regularization = float(self.config.hessian_regularization)
        self.iterations = int(self.config.iterations)
        self.relaxation = float(self.config.relaxation)
        self.use_constraint_clamp = bool(self.config.use_constraint_clamp)
        self.damping_compliance = float(self.config.damping_compliance)
        self.damping_beta = float(self.config.damping_beta)
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
        self.fsi_triangle_hydrostatic_support_scale = float(self.config.fsi_triangle_hydrostatic_support_scale)
        if self.fsi_triangle_hydrostatic_support_scale < 0.0:
            raise ValueError("IPBF triangle hydrostatic support scale must be non-negative.")
        self.fsi_triangle_reconstructed_support_depth_scale = float(
            self.config.fsi_triangle_reconstructed_support_depth_scale
        )
        if self.fsi_triangle_reconstructed_support_depth_scale < 0.0:
            raise ValueError("IPBF triangle reconstructed support depth scale must be non-negative.")
        self.fsi_static_boundary_weight = float(self.config.fsi_static_boundary_weight)
        if self.fsi_static_boundary_weight < 0.0:
            raise ValueError("IPBF static boundary weight must be non-negative.")
        self.fsi_decouple_triangle_boundary_density = bool(self.config.fsi_decouple_triangle_boundary_density)
        self.fsi_triangle_contact_enabled = bool(self.config.fsi_triangle_contact_enabled)
        self.fsi_triangle_contact_use_bvh = bool(self.config.fsi_triangle_contact_use_bvh)
        self.fsi_triangle_contact_use_grid = bool(self.config.fsi_triangle_contact_use_grid)
        configured_triangle_contact_search_radius = self.config.fsi_triangle_contact_search_radius
        self.fsi_triangle_contact_search_radius = (
            float(configured_triangle_contact_search_radius)
            if configured_triangle_contact_search_radius is not None
            else 0.0
        )
        if self.fsi_triangle_contact_search_radius < 0.0:
            raise ValueError("IPBF triangle contact search radius must be non-negative.")
        self.fsi_triangle_contact_continuous_enabled = bool(self.config.fsi_triangle_contact_continuous_enabled)
        self.fsi_triangle_contact_margin = float(self.config.fsi_triangle_contact_margin)
        if self.fsi_triangle_contact_margin < 0.0:
            raise ValueError("IPBF triangle contact margin must be non-negative.")
        self.fsi_triangle_contact_relaxation = float(self.config.fsi_triangle_contact_relaxation)
        if self.fsi_triangle_contact_relaxation < 0.0:
            raise ValueError("IPBF triangle contact relaxation must be non-negative.")
        self.fluid_particle_start, self.fluid_particle_count = self._validate_fluid_particle_range()
        self.fsi_triangle_contact_pair_capacity = self._resolve_triangle_contact_pair_capacity()
        self.fsi_triangle_contact_pair_cache_skin = self._resolve_triangle_contact_pair_cache_skin()
        self._uses_particle_subset = (
            self.fluid_particle_start != 0 or self.fluid_particle_count != self.model.particle_count
        )
        self._max_particle_radius = self._compute_max_particle_radius()
        self.boundary_model = None
        self.set_boundary_model(boundary_model)

        if not hasattr(model, "ipbf"):
            raise ValueError(
                "SolverIPBF requires IPBF custom attributes. "
                "Call SolverIPBF.register_custom_attributes(builder) before finalize()."
            )

        model.ipbf.rest_density.fill_(self.rest_density)
        model.ipbf.smoothing_radius.fill_(self.smoothing_radius)
        model.ipbf.compliance.fill_(self.compliance)

        self._initial_state = model.state()
        with wp.ScopedDevice(model.device):
            self._empty_body_q = wp.empty(0, dtype=wp.transform)
            self._boundary_projected_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projected_contact_count = wp.zeros(model.particle_count, dtype=int)
            self._viscosity_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._xsph_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_density = wp.zeros(model.particle_count, dtype=float)
            self._triangle_boundary_density = wp.zeros(model.particle_count, dtype=float)
            self._boundary_neighbor_count = wp.zeros(model.particle_count, dtype=wp.int32)
            self._boundary_projection_x_before = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projection_x_after = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projection_delta = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projection_delta_total = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._triangle_contact_particle_delta = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._triangle_contact_vertex_delta = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._triangle_contact_pair_particle = wp.full(
                self.fsi_triangle_contact_pair_capacity,
                value=-1,
                dtype=wp.int32,
            )
            self._triangle_contact_pair_triangle = wp.full(
                self.fsi_triangle_contact_pair_capacity,
                value=-1,
                dtype=wp.int32,
            )
            self._triangle_contact_pair_count = wp.zeros(1, dtype=wp.int32)
            self._triangle_contact_pair_overflow = wp.zeros(1, dtype=wp.int32)
            self._triangle_contact_pair_cache_valid = wp.zeros(1, dtype=wp.int32)
            self._triangle_contact_pair_cache_reuse = wp.zeros(1, dtype=wp.int32)
            self._triangle_contact_pair_cache_displacement_max = wp.zeros(1, dtype=float)
            self._triangle_contact_pair_cache_particle_q = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._triangle_contact_pair_cached_particle_mask = wp.zeros(model.particle_count, dtype=wp.int32)
            boundary_sample_count = int(getattr(self.boundary_model, "sample_count", 0)) if self.boundary_model else 0
            self._triangle_hydro_occupancy = wp.zeros(boundary_sample_count, dtype=float)
            self._triangle_hydro_surface_height = wp.zeros(boundary_sample_count, dtype=float)
            self._ipbf_particle_flags = wp.empty(model.particle_count, dtype=wp.int32)
            self._refresh_particle_flags()

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    def set_boundary_model(self, boundary_model: FSIBoundaryModel | None) -> None:
        """Set the optional FSI boundary sample model used by density assembly.

        Args:
            boundary_model: Boundary sample model, or ``None`` to disable
                solid-boundary density contributions.
        """
        if boundary_model is not None:
            if getattr(boundary_model, "model", self.model) is not self.model:
                raise ValueError("IPBF boundary model must be built from this solver's model.")
            if getattr(boundary_model, "device", self.model.device) != self.model.device:
                raise ValueError("IPBF boundary model device must match the solver model device.")

        self.boundary_model = boundary_model
        if hasattr(self, "_triangle_hydro_occupancy"):
            boundary_sample_count = int(getattr(boundary_model, "sample_count", 0)) if boundary_model else 0
            with wp.ScopedDevice(self.model.device):
                self._triangle_hydro_occupancy = wp.zeros(boundary_sample_count, dtype=float)
                self._triangle_hydro_surface_height = wp.zeros(boundary_sample_count, dtype=float)

    def _validate_fluid_particle_range(self) -> tuple[int, int]:
        """Return the validated consecutive particle range solved by IPBF."""
        model_particle_count = int(self.model.particle_count)
        start = int(self.config.fluid_particle_start)
        if start < 0 or start > model_particle_count:
            raise ValueError("IPBF fluid particle start must be within the model particle range.")

        configured_count = self.config.fluid_particle_count
        if configured_count is None:
            count = model_particle_count - start
        else:
            count = int(configured_count)

        if count < 0 or start + count > model_particle_count:
            raise ValueError("IPBF fluid particle count must fit inside the model particle range.")

        return start, count

    def _resolve_triangle_contact_pair_capacity(self) -> int:
        """Return the compact BVH triangle-contact pair capacity."""
        configured_capacity = self.config.fsi_triangle_contact_pair_capacity
        if configured_capacity is not None:
            capacity = int(configured_capacity)
            if capacity <= 0:
                raise ValueError("IPBF triangle contact pair capacity must be positive.")
            return capacity

        return max(256, max(1, self.fluid_particle_count) * 8)

    def _resolve_triangle_contact_pair_cache_skin(self) -> float:
        """Return the conservative motion skin used for compact pair cache reuse."""
        configured_skin = self.config.fsi_triangle_contact_pair_cache_skin
        if configured_skin is not None:
            skin = float(configured_skin)
            if skin < 0.0:
                raise ValueError("IPBF triangle contact pair cache skin must be non-negative.")
            return skin

        return max(1.0e-6, 0.5 * self._compute_max_particle_radius())

    def _compute_max_particle_radius(self) -> float:
        """Return the largest model particle radius [m]."""
        if self.model.particle_count == 0 or self.model.particle_radius is None:
            return 0.0

        return float(np.max(self.model.particle_radius.numpy()))

    def _refresh_particle_flags(self) -> None:
        """Refresh solver-local flags that mark only the IPBF fluid range active."""
        if self.model.particle_count == 0:
            return

        wp.launch(
            mask_ipbf_particle_flags,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_flags,
                self.fluid_particle_start,
                self.fluid_particle_count,
            ],
            outputs=[self._ipbf_particle_flags],
            device=self.model.device,
        )

    def _restore_non_fluid_particle_state(self, state_in: State, state_out: State) -> None:
        """Keep particles outside the IPBF fluid range untouched by the solver."""
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
        """Reset a state to the solver's initial particle configuration.

        Args:
            state_out: State to overwrite with the initial particle positions [m],
                velocities [m/s], zeroed forces [N], and initialized IPBF
                scratch buffers.
        """
        if state_out.particle_q is None or state_out.particle_qd is None or state_out.particle_f is None:
            raise ValueError("SolverIPBF.reset() requires a writable particle state.")

        if not hasattr(state_out, "ipbf"):
            raise ValueError(
                "State is missing IPBF attributes. Rebuild the model with SolverIPBF.register_custom_attributes()."
            )

        state_out.assign(self._initial_state)
        state_out.ipbf.y.assign(state_out.particle_q)

        wp.launch(
            initialize_guess_positions,
            dim=self.model.particle_count,
            inputs=[state_out.ipbf.y],
            outputs=[state_out.ipbf.x_guess, state_out.ipbf.x_new],
            device=self.model.device,
        )
        state_out.ipbf.x_star.assign(state_out.particle_q)

        state_out.ipbf.density.zero_()
        state_out.ipbf.neighbor_count.zero_()
        state_out.ipbf.constraint.zero_()
        state_out.ipbf.constraint_gradient.zero_()
        state_out.ipbf.force.zero_()
        state_out.ipbf.delta_q.zero_()
        self._boundary_density.zero_()
        self._triangle_boundary_density.zero_()
        self._boundary_neighbor_count.zero_()
        self._boundary_projection_x_before.zero_()
        self._boundary_projection_x_after.zero_()
        self._boundary_projection_delta.zero_()
        self._boundary_projection_delta_total.zero_()
        self._triangle_contact_particle_delta.zero_()
        self._triangle_contact_vertex_delta.zero_()
        self._triangle_contact_pair_count.zero_()
        self._triangle_contact_pair_overflow.zero_()
        self._triangle_contact_pair_cache_valid.zero_()
        self._triangle_contact_pair_cache_reuse.zero_()
        self._triangle_contact_pair_cache_displacement_max.zero_()
        self._triangle_contact_pair_cache_particle_q.zero_()
        self._triangle_contact_pair_cached_particle_mask.zero_()

        if self.boundary_model is not None:
            self.boundary_model.clear_forces()
            self.boundary_model.reset_deformable_contact_history()

        if self.model.particle_count > 1 and self.model.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.model.particle_grid.build(state_out.ipbf.x_guess, radius=self.smoothing_radius)

    def _has_shape_boundary_contacts(self, contacts: Contacts | None) -> bool:
        """Return whether shape boundary projection can be applied."""
        return contacts is not None and self.model.shape_count > 0 and contacts.soft_contact_max > 0

    def _has_triangle_boundary_contacts(self) -> bool:
        """Return whether deformable triangle boundary projection can be applied."""
        boundary_model = self.boundary_model
        return (
            self.fsi_triangle_contact_enabled
            and boundary_model is not None
            and self.model.tri_count > 0
            and self.model.tri_indices is not None
            and getattr(boundary_model, "triangle_count", 0) > 0
        )

    def _has_fsi_body_reaction_target(self, state: State) -> bool:
        """Return whether FSI body reaction accumulation is available."""
        boundary_model = self.boundary_model
        return (
            boundary_model is not None
            and self.model.body_count > 0
            and state.body_q is not None
            and self.model.body_com is not None
            and getattr(boundary_model, "shape_sample_count", None) is not None
        )

    def _apply_shape_boundary_contacts(
        self,
        state: State,
        particle_q: wp.array(dtype=wp.vec3),
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Project the given particle positions out of solid boundary contacts."""
        has_shape_contacts = self._has_shape_boundary_contacts(contacts)
        has_triangle_contacts = self._has_triangle_boundary_contacts()
        if not has_shape_contacts and not has_triangle_contacts:
            return

        model = self.model
        state.particle_q.assign(particle_q)

        boundary_model = self.boundary_model
        has_reaction_target = self._has_fsi_body_reaction_target(state)

        if has_shape_contacts:
            model.collide(state, contacts)

        if has_shape_contacts and has_reaction_target:
            self._boundary_projection_x_before.assign(state.particle_q)
            self._boundary_projection_delta.zero_()

            wp.launch(
                project_particle_shape_contacts_with_reaction,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_q,
                    model.particle_mass,
                    model.particle_radius,
                    self._ipbf_particle_flags,
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

            self._boundary_projection_x_after.assign(state.particle_q)
        elif has_shape_contacts:
            wp.launch(
                project_particle_shape_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    self._ipbf_particle_flags,
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

        if has_triangle_contacts:
            boundary_model.update_world_kinematics(state)
            boundary_model.build_grid()

            self._triangle_contact_particle_delta.zero_()
            self._triangle_contact_vertex_delta.zero_()
            triangle_contact_margin = float(self.fsi_triangle_contact_margin)
            if triangle_contact_margin == 0.0:
                triangle_contact_margin = 0.5 * float(getattr(boundary_model, "deformable_sample_thickness", 0.0))

            if (
                self.fsi_triangle_contact_use_bvh
                and getattr(boundary_model, "triangle_contact_bvh", None) is not None
            ):
                wp.launch(
                    initialize_triangle_contact_pair_cache_reuse,
                    dim=1,
                    inputs=[
                        self._triangle_contact_pair_cache_valid,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_reuse,
                        self._triangle_contact_pair_cache_displacement_max,
                    ],
                    device=model.device,
                )
                wp.launch(
                    clear_triangle_contact_pair_cache_reuse_for_triangle_count_change,
                    dim=1,
                    inputs=[
                        self._triangle_contact_pair_count,
                        self._triangle_contact_pair_cache_valid,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    device=model.device,
                )
                self._triangle_contact_pair_cached_particle_mask.zero_()
                wp.launch(
                    mark_triangle_contact_pair_cached_particles,
                    dim=self.fsi_triangle_contact_pair_capacity,
                    inputs=[
                        self._triangle_contact_pair_particle,
                        self._triangle_contact_pair_count,
                        self.fsi_triangle_contact_pair_capacity,
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cached_particle_mask,
                    ],
                    device=model.device,
                )
                wp.launch(
                    clear_triangle_contact_pair_cache_reuse_for_inactive,
                    dim=self.fsi_triangle_contact_pair_capacity,
                    inputs=[
                        self._ipbf_particle_flags,
                        self._triangle_contact_pair_particle,
                        self._triangle_contact_pair_count,
                        self.fsi_triangle_contact_pair_capacity,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    device=model.device,
                )
                wp.launch(
                    clear_triangle_contact_pair_cache_reuse_for_uncached_particles_near_bvh,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.triangle_contact_bvh.id,
                        self._triangle_contact_pair_cache_particle_q,
                        state.particle_q,
                        model.particle_radius,
                        self._ipbf_particle_flags,
                        self._triangle_contact_pair_cached_particle_mask,
                        self.fsi_triangle_contact_margin,
                        self.fsi_triangle_contact_pair_cache_skin,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    device=model.device,
                )
                wp.launch(
                    accumulate_triangle_contact_pair_cache_fluid_displacement_max,
                    dim=self.fsi_triangle_contact_pair_capacity,
                    inputs=[
                        self._triangle_contact_pair_particle,
                        self._triangle_contact_pair_count,
                        self.fsi_triangle_contact_pair_capacity,
                        state.particle_q,
                        self._triangle_contact_pair_cache_particle_q,
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_displacement_max,
                    ],
                    device=model.device,
                )
                wp.launch(
                    accumulate_triangle_contact_pair_cache_triangle_displacement_max,
                    dim=boundary_model.triangle_count,
                    inputs=[
                        boundary_model.triangle_indices,
                        model.tri_indices,
                        state.particle_q,
                        self._triangle_contact_pair_cache_particle_q,
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_displacement_max,
                    ],
                    device=model.device,
                )
                wp.launch(
                    finalize_triangle_contact_pair_cache_reuse,
                    dim=1,
                    inputs=[
                        self._triangle_contact_pair_cache_valid,
                        self._triangle_contact_pair_count,
                        self._triangle_contact_pair_overflow,
                        self._triangle_contact_pair_cache_displacement_max,
                        self.fsi_triangle_contact_pair_cache_skin,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    device=model.device,
                )
                wp.launch(
                    prepare_triangle_contact_pair_collection,
                    dim=1,
                    inputs=[
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_count,
                        self._triangle_contact_pair_overflow,
                    ],
                    device=model.device,
                )
                wp.launch(
                    collect_particle_triangle_contact_pairs_from_bvh,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.triangle_contact_bvh.id,
                        boundary_model.triangle_indices,
                        boundary_model.triangle_contact_particle_q_prev,
                        state.particle_q,
                        model.particle_radius,
                        self._ipbf_particle_flags,
                        triangle_contact_margin,
                        self._triangle_contact_pair_cache_reuse,
                        self.fsi_triangle_contact_pair_cache_skin,
                        self.fsi_triangle_contact_pair_capacity,
                    ],
                    outputs=[
                        self._triangle_contact_pair_particle,
                        self._triangle_contact_pair_triangle,
                        self._triangle_contact_pair_count,
                        self._triangle_contact_pair_overflow,
                    ],
                    device=model.device,
                )
                wp.launch(
                    update_triangle_contact_pair_cache_snapshot_if_active,
                    dim=model.particle_count,
                    inputs=[
                        self._ipbf_particle_flags,
                        state.particle_q,
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_particle_q,
                    ],
                    device=model.device,
                )
                wp.launch(
                    update_triangle_contact_pair_cache_snapshot_for_sampled_triangles,
                    dim=boundary_model.triangle_count,
                    inputs=[
                        boundary_model.triangle_indices,
                        model.tri_indices,
                        state.particle_q,
                        self._triangle_contact_pair_cache_reuse,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_particle_q,
                    ],
                    device=model.device,
                )
                wp.launch(
                    finalize_triangle_contact_pair_cache_collection,
                    dim=1,
                    inputs=[
                        self._triangle_contact_pair_cache_reuse,
                        self._triangle_contact_pair_overflow,
                    ],
                    outputs=[
                        self._triangle_contact_pair_cache_valid,
                    ],
                    device=model.device,
                )
                wp.launch(
                    accumulate_particle_triangle_contact_corrections_from_pairs,
                    dim=self.fsi_triangle_contact_pair_capacity,
                    inputs=[
                        self._triangle_contact_pair_particle,
                        self._triangle_contact_pair_triangle,
                        self._triangle_contact_pair_count,
                        self._triangle_contact_pair_overflow,
                        self.fsi_triangle_contact_pair_capacity,
                        boundary_model.triangle_contact_particle_q_prev,
                        state.particle_q,
                        model.particle_mass,
                        model.particle_inv_mass,
                        model.particle_radius,
                        model.tri_indices,
                        triangle_contact_margin,
                        self.fsi_triangle_contact_relaxation,
                        int(self.fsi_triangle_contact_continuous_enabled),
                        dt,
                    ],
                    outputs=[
                        self._triangle_contact_particle_delta,
                        self._triangle_contact_vertex_delta,
                        boundary_model.vertex_contact_delta,
                        boundary_model.vertex_force,
                    ],
                    device=model.device,
                )
                wp.launch(
                    accumulate_particle_triangle_contact_corrections_if_overflow,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.triangle_contact_particle_q_prev,
                        state.particle_q,
                        model.particle_mass,
                        model.particle_inv_mass,
                        model.particle_radius,
                        self._ipbf_particle_flags,
                        model.tri_indices,
                        boundary_model.triangle_indices,
                        boundary_model.triangle_count,
                        self._triangle_contact_pair_overflow,
                        triangle_contact_margin,
                        self.fsi_triangle_contact_relaxation,
                        int(self.fsi_triangle_contact_continuous_enabled),
                        dt,
                    ],
                    outputs=[
                        self._triangle_contact_particle_delta,
                        self._triangle_contact_vertex_delta,
                        boundary_model.vertex_contact_delta,
                        boundary_model.vertex_force,
                    ],
                    device=model.device,
                )
            elif (
                self.fsi_triangle_contact_use_grid
                and getattr(boundary_model, "triangle_contact_grid", None) is not None
            ):
                contact_search_radius = max(
                    self.fsi_triangle_contact_search_radius,
                    float(getattr(boundary_model, "triangle_contact_radius_max", 0.0))
                    + self._max_particle_radius
                    + triangle_contact_margin,
                )
                wp.launch(
                    accumulate_particle_triangle_contact_corrections_from_grid,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.triangle_contact_grid.id,
                        boundary_model.triangle_contact_particle_q_prev,
                        state.particle_q,
                        model.particle_mass,
                        model.particle_inv_mass,
                        model.particle_radius,
                        self._ipbf_particle_flags,
                        model.tri_indices,
                        boundary_model.triangle_indices,
                        boundary_model.triangle_contact_proxy_motion_max,
                        contact_search_radius,
                        triangle_contact_margin,
                        self.fsi_triangle_contact_relaxation,
                        int(self.fsi_triangle_contact_continuous_enabled),
                        dt,
                    ],
                    outputs=[
                        self._triangle_contact_particle_delta,
                        self._triangle_contact_vertex_delta,
                        boundary_model.vertex_contact_delta,
                        boundary_model.vertex_force,
                    ],
                    device=model.device,
                )
            else:
                wp.launch(
                    accumulate_particle_triangle_contact_corrections,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.triangle_contact_particle_q_prev,
                        state.particle_q,
                        model.particle_mass,
                        model.particle_inv_mass,
                        model.particle_radius,
                        self._ipbf_particle_flags,
                        model.tri_indices,
                        boundary_model.triangle_indices,
                        boundary_model.triangle_count,
                        triangle_contact_margin,
                        self.fsi_triangle_contact_relaxation,
                        int(self.fsi_triangle_contact_continuous_enabled),
                        dt,
                    ],
                    outputs=[
                        self._triangle_contact_particle_delta,
                        self._triangle_contact_vertex_delta,
                        boundary_model.vertex_contact_delta,
                        boundary_model.vertex_force,
                    ],
                    device=model.device,
                )

            wp.launch(
                apply_particle_triangle_contact_deltas,
                dim=model.particle_count,
                inputs=[
                    self._ipbf_particle_flags,
                    self._triangle_contact_particle_delta,
                    self._triangle_contact_vertex_delta,
                    # In subset / FSI mode the deformable boundary is owned by
                    # the solid solver, so IPBF should only move the fluid side
                    # and leave cloth vertex motion to VBD.
                    int(not self._uses_particle_subset),
                    state.particle_q,
                ],
                device=model.device,
            )

        particle_q.assign(state.particle_q)

    def _apply_shape_boundary_velocity_projection(self, state: State, contacts: Contacts | None, dt: float) -> None:
        """Project final particle velocities against active particle-shape contacts."""
        if not self._has_shape_boundary_contacts(contacts):
            return

        model = self.model
        model.collide(state, contacts)

        wp.launch(
            initialize_particle_shape_boundary_velocity_projection,
            dim=model.particle_count,
            inputs=[
                self._ipbf_particle_flags,
            ],
            outputs=[
                self._boundary_projected_particle_qd,
                self._boundary_projected_contact_count,
            ],
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
                    self._ipbf_particle_flags,
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
                    self._ipbf_particle_flags,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    self.boundary_velocity_damping,
                ],
                outputs=[
                    self._boundary_projected_particle_qd,
                    self._boundary_projected_contact_count,
                ],
                device=model.device,
            )

        wp.launch(
            finalize_particle_shape_boundary_velocity_projection,
            dim=model.particle_count,
            inputs=[
                state.particle_qd,
                self._ipbf_particle_flags,
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
                        state.ipbf.density,
                        model.particle_mass,
                        self._ipbf_particle_flags,
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
                        state.ipbf.density,
                        model.particle_mass,
                        self._ipbf_particle_flags,
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
                    state.ipbf.density,
                    self._ipbf_particle_flags,
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
                        state.ipbf.density,
                        model.particle_mass,
                        self._ipbf_particle_flags,
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
                        state.ipbf.density,
                        model.particle_mass,
                        self._ipbf_particle_flags,
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
                        state.ipbf.density,
                        self._ipbf_particle_flags,
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
                        self._ipbf_particle_flags,
                        self.viscosity_coefficient,
                        dt,
                    ],
                    outputs=[self._viscosity_particle_qd],
                    device=model.device,
                )

        state.particle_qd.assign(self._viscosity_particle_qd)

    def _compute_iteration_fields(
        self,
        state: State,
        particle_q: wp.array(dtype=wp.vec3),
        dt: float,
        *,
        recompute_density_constraint: bool = True,
        compliance: float | None = None,
        boundary_model: FSIBoundaryModel | None = None,
    ) -> None:
        """Assemble IPBF local solve quantities for the given particle positions.

        Args:
            state: State whose ``ipbf`` buffers receive the assembled quantities.
            particle_q: Current particle positions [m] used to build the local
                IPBF system.
            dt: Timestep size [s].
            recompute_density_constraint: Whether to recompute density and
                constraint terms on ``particle_q`` before assembling the local
                linear system.
            compliance: Compliance override used when assembling force and
                Hessian terms. If ``None``, uses the solver's default
                compliance.
            boundary_model: Optional FSI boundary sample model whose samples
                contribute to fluid density and constraint gradients.
        """
        model = self.model
        compliance_value = self.compliance if compliance is None else float(compliance)
        boundary_model = self.boundary_model if boundary_model is None else boundary_model
        has_boundary = (
            boundary_model is not None
            and getattr(boundary_model, "sample_count", 0) > 0
            and getattr(boundary_model, "boundary_grid", None) is not None
        )

        if recompute_density_constraint:
            self._boundary_density.zero_()
            self._triangle_boundary_density.zero_()
            self._boundary_neighbor_count.zero_()

        if recompute_density_constraint and has_boundary:
            boundary_model.update_world_kinematics(state)
            boundary_model.build_grid()

        if recompute_density_constraint and model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(particle_q, radius=self.smoothing_radius)

            if has_boundary:
                wp.launch(
                    update_boundary_triangle_wet_weights,
                    dim=boundary_model.sample_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        self._ipbf_particle_flags,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_normal_world,
                        self.smoothing_radius,
                    ],
                    outputs=[boundary_model.sample_wet_weight],
                    device=model.device,
                )

            if has_boundary:
                wp.launch(
                    compute_density_and_neighbor_count_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._ipbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_normal_world,
                        boundary_model.sample_wet_weight,
                        boundary_model.sample_vertex0,
                        boundary_model.sample_vertex1,
                        boundary_model.sample_vertex2,
                        boundary_model.sample_barycentric,
                        boundary_model.sample_area_patch,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self._triangle_contact_vertex_delta,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.fsi_triangle_reconstructed_support_depth_scale,
                    ],
                    outputs=[
                        state.ipbf.density,
                        state.ipbf.neighbor_count,
                        self._boundary_density,
                        self._triangle_boundary_density,
                        self._boundary_neighbor_count,
                    ],
                    device=model.device,
                )

                wp.launch(
                    compute_constraint_and_gradient_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._ipbf_particle_flags,
                        model.particle_world,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_normal_world,
                        boundary_model.sample_wet_weight,
                        boundary_model.sample_vertex0,
                        boundary_model.sample_vertex1,
                        boundary_model.sample_vertex2,
                        boundary_model.sample_barycentric,
                        boundary_model.sample_area_patch,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self._triangle_contact_vertex_delta,
                        self.fsi_static_boundary_weight,
                        state.ipbf.density,
                        self._triangle_boundary_density,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        int(self.use_constraint_clamp),
                        int(self.fsi_decouple_triangle_boundary_density),
                        self.fsi_triangle_reconstructed_support_depth_scale,
                    ],
                    outputs=[state.ipbf.constraint, state.ipbf.constraint_gradient],
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
                        self._ipbf_particle_flags,
                        model.particle_world,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[state.ipbf.density, state.ipbf.neighbor_count],
                    device=model.device,
                )

                wp.launch(
                    compute_constraint_and_gradient,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._ipbf_particle_flags,
                        model.particle_world,
                        state.ipbf.density,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        int(self.use_constraint_clamp),
                    ],
                    outputs=[state.ipbf.constraint, state.ipbf.constraint_gradient],
                    device=model.device,
                )
        elif recompute_density_constraint:
            if has_boundary:
                wp.launch(
                    initialize_density_and_neighbor_count_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.boundary_grid.id,
                        particle_q,
                        model.particle_mass,
                        self._ipbf_particle_flags,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_normal_world,
                        boundary_model.sample_wet_weight,
                        boundary_model.sample_vertex0,
                        boundary_model.sample_vertex1,
                        boundary_model.sample_vertex2,
                        boundary_model.sample_barycentric,
                        boundary_model.sample_area_patch,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self._triangle_contact_vertex_delta,
                        self.fsi_static_boundary_weight,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        self.fsi_triangle_reconstructed_support_depth_scale,
                    ],
                    outputs=[
                        state.ipbf.density,
                        state.ipbf.neighbor_count,
                        self._boundary_density,
                        self._triangle_boundary_density,
                        self._boundary_neighbor_count,
                    ],
                    device=model.device,
                )

                wp.launch(
                    initialize_constraint_and_gradient_with_boundary,
                    dim=model.particle_count,
                    inputs=[
                        boundary_model.boundary_grid.id,
                        particle_q,
                        self._ipbf_particle_flags,
                        boundary_model.sample_x_world,
                        boundary_model.sample_triangle,
                        boundary_model.sample_normal_world,
                        boundary_model.sample_wet_weight,
                        boundary_model.sample_vertex0,
                        boundary_model.sample_vertex1,
                        boundary_model.sample_vertex2,
                        boundary_model.sample_barycentric,
                        boundary_model.sample_area_patch,
                        boundary_model.sample_volume_hydrostatic,
                        boundary_model.sample_flags,
                        self._triangle_contact_vertex_delta,
                        self.fsi_static_boundary_weight,
                        state.ipbf.density,
                        self._triangle_boundary_density,
                        self.rest_density,
                        self.smoothing_radius,
                        self.kernel_family,
                        int(self.use_constraint_clamp),
                        int(self.fsi_decouple_triangle_boundary_density),
                        self.fsi_triangle_reconstructed_support_depth_scale,
                    ],
                    outputs=[state.ipbf.constraint, state.ipbf.constraint_gradient],
                    device=model.device,
                )
            else:
                wp.launch(
                    initialize_density_and_neighbor_count,
                    dim=model.particle_count,
                    inputs=[
                        particle_q,
                        model.particle_mass,
                        self._ipbf_particle_flags,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[state.ipbf.density, state.ipbf.neighbor_count],
                    device=model.device,
                )

                wp.launch(
                    initialize_constraint_and_gradient,
                    dim=model.particle_count,
                    inputs=[
                        state.ipbf.density,
                        self._ipbf_particle_flags,
                        self.rest_density,
                        int(self.use_constraint_clamp),
                    ],
                    outputs=[state.ipbf.constraint, state.ipbf.constraint_gradient],
                    device=model.device,
                )

        if model.particle_count > 1 and model.particle_grid is not None:
            wp.launch(
                compute_force,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    particle_q,
                    state.ipbf.y,
                    model.particle_mass,
                    self._ipbf_particle_flags,
                    model.particle_world,
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    compliance_value,
                    dt,
                ],
                outputs=[state.ipbf.force],
                device=model.device,
            )

            wp.launch(
                compute_hessian,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    particle_q,
                    model.particle_mass,
                    self._ipbf_particle_flags,
                    model.particle_world,
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    compliance_value,
                    dt,
                    self.hessian_regularization,
                ],
                outputs=[state.ipbf.hessian],
                device=model.device,
            )
        else:
            wp.launch(
                compute_force_without_grid,
                dim=model.particle_count,
                inputs=[
                    particle_q,
                    state.ipbf.y,
                    model.particle_mass,
                    self._ipbf_particle_flags,
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    compliance_value,
                    dt,
                ],
                outputs=[state.ipbf.force],
                device=model.device,
            )

            wp.launch(
                compute_hessian_without_grid,
                dim=model.particle_count,
                inputs=[
                    model.particle_mass,
                    self._ipbf_particle_flags,
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    compliance_value,
                    dt,
                    self.hessian_regularization,
                ],
                outputs=[state.ipbf.hessian],
                device=model.device,
            )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by one timestep.

        The current implementation follows the first iterative IPBF scaffold:

        1. Predict inertial target positions ``y``.
        2. Initialize ``x_guess`` and ``x_new`` from ``y``.
        3. Assemble density, constraint, force, and Hessian terms on ``x_guess``.
        4. Solve the local 3x3 system for ``delta_q``.
        5. Apply a relaxed Jacobi position update to form ``x_new``.
        6. Repeat for ``iterations`` rounds.
        7. During the final iteration, optionally compute a single alternative
           soft-compliance update for artificial damping.
        8. If particle-shape or triangle-boundary contacts are enabled,
           project iterates back out of penetrating solid boundaries after
           each update.
        9. Recompute the diagnostic fields on the final positions.
        10. Commit the converged positions and reconstruct velocities, applying
           the artificial damping correction when enabled.

        The current force and Hessian use a stable approximation that includes
        neighborhood Gauss-Newton terms and diagonalized second-order
        stabilization. Particle-shape contacts are handled by a minimal
        normal-direction positional projection using Newton soft contacts;
        triangle-boundary contacts use a symmetric PBD-style correction and
        store cloth-side deltas on the boundary model.
        """
        if not self._begin_step(state_in, state_out, dt):
            return

        if self.iterations <= 0:
            self._solve_zero_iteration(state_out, contacts, dt)
            self._finalize_step(state_in, state_out, contacts, dt, recompute_fields=False)
            return

        for iteration in range(self.iterations):
            self._solve_iteration(state_out, contacts, dt, iteration, self.iterations)

        self._finalize_step(state_in, state_out, contacts, dt)

    def _begin_step(self, state_in: State, state_out: State, dt: float) -> bool:
        """Prepare IPBF buffers for one timestep."""
        if state_in.particle_q is None or state_in.particle_qd is None:
            raise ValueError("SolverIPBF requires particle positions and velocities.")

        if state_in.particle_f is None or state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("SolverIPBF requires particle forces and writable output particle state.")

        if not hasattr(state_out, "ipbf"):
            raise ValueError(
                "State is missing IPBF attributes. Rebuild the model with SolverIPBF.register_custom_attributes()."
            )

        model = self.model
        if model.particle_count == 0 or self.fluid_particle_count == 0:
            return False
        self._refresh_particle_flags()

        if self.boundary_model is not None:
            self.boundary_model.clear_forces()
            self.boundary_model.initialize_deformable_contact_history(state_in)
            self._boundary_projection_delta_total.zero_()
            self._triangle_contact_particle_delta.zero_()
            self._triangle_contact_vertex_delta.zero_()

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
                self._ipbf_particle_flags,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[state_out.ipbf.y],
            device=model.device,
        )

        wp.launch(
            initialize_guess_positions,
            dim=model.particle_count,
            inputs=[state_out.ipbf.y],
            outputs=[state_out.ipbf.x_guess, state_out.ipbf.x_new],
            device=model.device,
        )
        state_out.ipbf.x_star.assign(state_out.ipbf.y)

        return True

    def _solve_zero_iteration(self, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """Apply projection-only IPBF work for zero-iteration configurations."""
        self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_guess, contacts, dt)
        state_out.ipbf.x_new.assign(state_out.ipbf.x_guess)
        state_out.ipbf.x_star.assign(state_out.ipbf.x_guess)
        self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)
        state_out.ipbf.delta_q.zero_()

    def _solve_iteration(
        self,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        iteration: int,
        iteration_count: int | None = None,
    ) -> None:
        """Run one relaxed Jacobi IPBF iteration."""
        model = self.model
        iteration_count = self.iterations if iteration_count is None else int(iteration_count)
        is_last_iteration = iteration == iteration_count - 1

        self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)

        if is_last_iteration and self.damping_beta > 0.0 and self.smoothing_radius > 0.0:
            self._compute_iteration_fields(
                state_out,
                state_out.ipbf.x_guess,
                dt,
                recompute_density_constraint=False,
                compliance=self.damping_compliance,
            )

            wp.launch(
                solve_local_system,
                dim=model.particle_count,
                inputs=[
                    self._ipbf_particle_flags,
                    state_out.ipbf.force,
                    state_out.ipbf.hessian,
                ],
                outputs=[state_out.ipbf.delta_q],
                device=model.device,
            )

            wp.launch(
                apply_relaxed_jacobi_update,
                dim=model.particle_count,
                inputs=[
                    self._ipbf_particle_flags,
                    state_out.ipbf.x_guess,
                    state_out.ipbf.delta_q,
                    self.relaxation,
                ],
                outputs=[state_out.ipbf.x_star],
                device=model.device,
            )
            self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_star, contacts, dt)

            self._compute_iteration_fields(
                state_out,
                state_out.ipbf.x_guess,
                dt,
                recompute_density_constraint=False,
                compliance=self.compliance,
            )
        elif is_last_iteration:
            state_out.ipbf.x_star.assign(state_out.ipbf.x_guess)

        self._accumulate_boundary_pressure_reaction(state_out, state_out.ipbf.x_guess, dt)

        wp.launch(
            solve_local_system,
            dim=model.particle_count,
            inputs=[
                self._ipbf_particle_flags,
                state_out.ipbf.force,
                state_out.ipbf.hessian,
            ],
            outputs=[state_out.ipbf.delta_q],
            device=model.device,
        )

        wp.launch(
            apply_relaxed_jacobi_update,
            dim=model.particle_count,
            inputs=[
                self._ipbf_particle_flags,
                state_out.ipbf.x_guess,
                state_out.ipbf.delta_q,
                self.relaxation,
            ],
            outputs=[state_out.ipbf.x_new],
            device=model.device,
        )
        self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_new, contacts, dt)

        state_out.ipbf.x_guess.assign(state_out.ipbf.x_new)

    def _accumulate_boundary_pressure_reaction(
        self,
        state: State,
        particle_q: wp.array(dtype=wp.vec3),
        dt: float,
    ) -> None:
        """Accumulate pressure-gradient FSI reaction from boundary samples."""
        boundary_model = self.boundary_model
        if (
            boundary_model is None
            or self.fsi_pressure_reaction_relaxation == 0.0
            or getattr(boundary_model, "sample_count", 0) == 0
            or getattr(boundary_model, "boundary_grid", None) is None
        ):
            return

        model = self.model
        wp.launch(
            accumulate_boundary_pressure_reaction,
            dim=model.particle_count,
            inputs=[
                boundary_model.boundary_grid.id,
                particle_q,
                model.particle_mass,
                self._ipbf_particle_flags,
                state.ipbf.density,
                state.ipbf.constraint,
                state.ipbf.hessian,
                boundary_model.sample_x_world,
                boundary_model.sample_body,
                boundary_model.sample_triangle,
                boundary_model.sample_normal_world,
                boundary_model.sample_wet_weight,
                boundary_model.sample_vertex0,
                boundary_model.sample_vertex1,
                boundary_model.sample_vertex2,
                boundary_model.sample_barycentric,
                boundary_model.sample_area_patch,
                boundary_model.sample_volume_hydrostatic,
                boundary_model.sample_flags,
                self._triangle_boundary_density,
                state.body_q if state.body_q is not None else boundary_model._empty_body_q,
                model.body_com if model.body_com is not None else boundary_model._empty_body_com,
                self.rest_density,
                self.smoothing_radius,
                self.kernel_family,
                self.fsi_static_boundary_weight,
                dt,
                self.relaxation,
                self.fsi_pressure_reaction_relaxation,
                self.fsi_triangle_reconstructed_support_depth_scale,
                int(self.fsi_decouple_triangle_boundary_density),
                self._triangle_contact_vertex_delta,
            ],
            outputs=[
                boundary_model.vertex_contact_delta,
                boundary_model.sample_force,
                boundary_model.vertex_force,
                boundary_model.vertex_pressure_force,
                boundary_model.body_force,
                boundary_model.body_torque,
            ],
            device=model.device,
        )

        if (
            self.fsi_triangle_hydrostatic_support_scale == 0.0
            or boundary_model.sample_count == 0
            or model.particle_grid is None
        ):
            return

        self._triangle_hydro_occupancy.zero_()
        self._triangle_hydro_surface_height.zero_()

        wp.launch(
            accumulate_boundary_triangle_hydrostatic_state,
            dim=boundary_model.sample_count,
            inputs=[
                model.particle_grid.id,
                particle_q,
                model.particle_mass,
                self._ipbf_particle_flags,
                model.particle_world,
                boundary_model.sample_x_world,
                boundary_model.sample_triangle,
                boundary_model.sample_normal_world,
                boundary_model.sample_wet_weight,
                boundary_model.sample_vertex0,
                boundary_model.sample_flags,
                model.gravity,
                self.rest_density,
                self.smoothing_radius,
                self.kernel_family,
                self.fsi_static_boundary_weight,
            ],
            outputs=[
                self._triangle_hydro_occupancy,
                self._triangle_hydro_surface_height,
            ],
            device=model.device,
        )

        wp.launch(
            apply_boundary_triangle_hydrostatic_support,
            dim=boundary_model.sample_count,
            inputs=[
                boundary_model.sample_x_world,
                boundary_model.sample_body,
                boundary_model.sample_triangle,
                boundary_model.sample_normal_world,
                boundary_model.sample_wet_weight,
                boundary_model.sample_vertex0,
                boundary_model.sample_vertex1,
                boundary_model.sample_vertex2,
                boundary_model.sample_barycentric,
                boundary_model.sample_area_patch,
                boundary_model.sample_flags,
                model.particle_inv_mass,
                model.particle_world,
                self._triangle_contact_vertex_delta,
                state.body_q if state.body_q is not None else boundary_model._empty_body_q,
                model.body_com if model.body_com is not None else boundary_model._empty_body_com,
                model.gravity,
                self.rest_density,
                dt,
                self.fsi_triangle_hydrostatic_support_scale,
                self._triangle_hydro_occupancy,
                self._triangle_hydro_surface_height,
            ],
            outputs=[
                boundary_model.sample_force,
                boundary_model.vertex_contact_delta,
                boundary_model.vertex_force,
                boundary_model.vertex_pressure_force,
                boundary_model.body_force,
                boundary_model.body_torque,
            ],
            device=model.device,
        )

    def _finalize_step(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        *,
        recompute_fields: bool = True,
    ) -> None:
        """Commit converged IPBF positions and reconstruct particle velocities."""
        model = self.model

        if recompute_fields:
            self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)

        state_out.particle_q.assign(state_out.ipbf.x_guess)
        if self.iterations > 0 and self.damping_beta > 0.0 and self.smoothing_radius > 0.0:
            wp.launch(
                apply_artificial_damping,
                dim=model.particle_count,
                inputs=[
                    state_out.ipbf.x_guess,
                    state_out.ipbf.x_star,
                    state_in.particle_q,
                    self._ipbf_particle_flags,
                    dt,
                    self.smoothing_radius,
                    self.damping_beta,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )
        else:
            state_out.ipbf.x_star.assign(state_out.ipbf.x_guess)
            wp.launch(
                update_velocity_from_positions,
                dim=model.particle_count,
                inputs=[
                    state_out.ipbf.x_guess,
                    state_in.particle_q,
                    self._ipbf_particle_flags,
                    dt,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )

        self._apply_shape_boundary_velocity_projection(state_out, contacts, dt)
        self._apply_viscosity_velocity_diffusion(state_out, dt)
        self._apply_xsph_velocity_smoothing(state_out)
        self._restore_non_fluid_particle_state(state_in, state_out)
