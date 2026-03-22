"""Implicit Position-Based Fluids solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

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
    accumulate_particle_shape_boundary_velocity_projection,
    apply_artificial_damping,
    apply_relaxed_jacobi_update,
    apply_viscosity_velocity_diffusion,
    apply_viscosity_velocity_diffusion_without_grid,
    apply_xsph_velocity_smoothing,
    apply_xsph_velocity_smoothing_without_grid,
    compute_boundary_particle_volumes,
    compute_constraint_and_gradient,
    compute_density_and_neighbor_count,
    compute_force,
    compute_force_without_grid,
    compute_hessian,
    compute_hessian_without_grid,
    finalize_particle_shape_boundary_velocity_projection,
    initialize_particle_shape_boundary_velocity_projection,
    initialize_guess_positions,
    initialize_constraint_and_gradient,
    initialize_density_and_neighbor_count,
    predict_inertial_positions,
    project_particle_shape_contacts,
    solve_local_system,
    update_velocity_from_positions,
)

__all__ = ["SolverIPBF"]


class SolverIPBF(SolverBase):
    """Implicit Position-Based Fluids solver.

    This solver implements an Implicit Position-Based Fluids algorithm,
    roughly following [1].

    [1] https://doi.org/10.1145/3757377.3764005

    Args:
        model: The model to solve.
        config: The solver configuration.

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
            boundary_mode: Boundary handling mode used by the solver.
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
            xsph_coefficient: XSPH velocity smoothing coefficient applied after
                the position solve.
            boundary_velocity_damping: Tangential damping multiplier applied to
                particle velocities at final particle-shape contacts.
        """

        class KernelFamily(IntEnum):
            """Selectable SPH kernel families for IPBF."""

            CUBIC_SPLINE = 0
            POLY6 = 1

        class BoundaryMode(IntEnum):
            """Selectable boundary handling modes for IPBF."""

            SHAPE_PROJECTION = 0
            BOUNDARY_PARTICLES = 1

        rest_density: float = 1000.0
        smoothing_radius: float = 0.1
        kernel_family: KernelFamily = KernelFamily.CUBIC_SPLINE
        boundary_mode: BoundaryMode = BoundaryMode.SHAPE_PROJECTION
        compliance: float = 0.0
        hessian_regularization: float = 1.0e-6
        iterations: int = 3
        relaxation: float = 0.5
        use_constraint_clamp: bool = True
        damping_compliance: float = 1.0 / 1000.0
        damping_beta: float = 60.0
        viscosity_coefficient: float = 0.0
        xsph_coefficient: float = 0.0
        boundary_velocity_damping: float = 1.0

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
            - ``ipbf:neighbor_count``: Number of active neighboring particles inside the support radius
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
    ):
        super().__init__(model)

        self.config = config if config is not None else self.Config()
        self.rest_density = float(self.config.rest_density)
        self.smoothing_radius = float(self.config.smoothing_radius)
        self.kernel_family = int(self.Config.KernelFamily(self.config.kernel_family))
        self.boundary_mode = int(self.Config.BoundaryMode(self.config.boundary_mode))
        self.compliance = float(self.config.compliance)
        self.hessian_regularization = float(self.config.hessian_regularization)
        self.iterations = int(self.config.iterations)
        self.relaxation = float(self.config.relaxation)
        self.use_constraint_clamp = bool(self.config.use_constraint_clamp)
        self.damping_compliance = float(self.config.damping_compliance)
        self.damping_beta = float(self.config.damping_beta)
        self.viscosity_coefficient = float(self.config.viscosity_coefficient)
        self.xsph_coefficient = float(self.config.xsph_coefficient)
        self.boundary_velocity_damping = float(self.config.boundary_velocity_damping)

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
            self._empty_vec3 = wp.empty(0, dtype=wp.vec3)
            self._empty_float = wp.empty(0, dtype=float)
            self._empty_int = wp.empty(0, dtype=int)
            self._boundary_projected_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._boundary_projected_contact_count = wp.zeros(model.particle_count, dtype=int)
            self._viscosity_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)
            self._xsph_particle_qd = wp.zeros(model.particle_count, dtype=wp.vec3)

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        self._boundary_particle_q = self._empty_vec3
        self._boundary_particle_volume = self._empty_float
        self._boundary_particle_world = self._empty_int
        self._boundary_particle_grid: wp.HashGrid | None = None
        self._boundary_particle_count = 0

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
                "State is missing IPBF attributes. "
                "Rebuild the model with SolverIPBF.register_custom_attributes()."
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

        if self.model.particle_count > 1 and self.model.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.model.particle_grid.build(state_out.ipbf.x_guess, radius=self.smoothing_radius)

    def clear_boundary_particles(self) -> None:
        """Clear all solver-owned static boundary particles."""
        self._boundary_particle_q = self._empty_vec3
        self._boundary_particle_volume = self._empty_float
        self._boundary_particle_world = self._empty_int
        self._boundary_particle_grid = None
        self._boundary_particle_count = 0

    def setup_boundary_particles(
        self,
        positions: np.ndarray | list[tuple[float, float, float]] | list[list[float]],
        *,
        volumes: np.ndarray | list[float] | None = None,
        world_indices: np.ndarray | list[int] | None = None,
        spacing: float | None = None,
    ) -> None:
        """Initialize static boundary particles owned by the solver.

        Args:
            positions: Boundary particle positions [m], shape [count, 3].
            volumes: Effective boundary particle volumes [m^3], shape [count].
                If omitted, ``spacing`` must be provided and a uniform
                ``spacing**3`` volume is used.
            world_indices: World index of each boundary particle. If omitted,
                all particles are assigned to world 0.
            spacing: Sampling spacing [m] used when ``volumes`` are omitted.
        """
        positions_np = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
        count = int(positions_np.shape[0])

        if count == 0:
            self.clear_boundary_particles()
            return

        if volumes is None:
            volumes_np = None
        else:
            volumes_np = np.asarray(volumes, dtype=np.float32).reshape((-1,))
            if volumes_np.shape[0] != count:
                raise ValueError("Boundary particle volume count must match the number of boundary particle positions.")

        if world_indices is None:
            world_np = np.zeros(count, dtype=np.int32)
        else:
            world_np = np.asarray(world_indices, dtype=np.int32).reshape((-1,))
            if world_np.shape[0] != count:
                raise ValueError("Boundary particle world-index count must match the number of boundary particle positions.")

        with wp.ScopedDevice(self.model.device):
            self._boundary_particle_q = wp.array(positions_np, dtype=wp.vec3, device=self.model.device)
            self._boundary_particle_world = wp.array(world_np, dtype=int, device=self.model.device)
            self._boundary_particle_grid = wp.HashGrid(128, 128, 128)
            self._boundary_particle_grid.reserve(count)
            self._boundary_particle_grid.build(self._boundary_particle_q, radius=self.smoothing_radius)
            if volumes_np is None:
                self._boundary_particle_volume = wp.zeros(count, dtype=float, device=self.model.device)
                wp.launch(
                    compute_boundary_particle_volumes,
                    dim=count,
                    inputs=[
                        self._boundary_particle_grid.id,
                        self._boundary_particle_q,
                        self._boundary_particle_world,
                        self.smoothing_radius,
                        self.kernel_family,
                    ],
                    outputs=[self._boundary_particle_volume],
                    device=self.model.device,
                )
            else:
                self._boundary_particle_volume = wp.array(volumes_np, dtype=float, device=self.model.device)

        self._boundary_particle_count = count

    def setup_boundary_particles_box(
        self,
        *,
        half_width: float,
        half_depth: float,
        wall_half_height: float,
        spacing: float,
        floor_y: float = 0.0,
        world_index: int = 0,
    ) -> None:
        """Sample a static open-top box boundary into boundary particles.

        Args:
            half_width: Interior half-width of the box [m].
            half_depth: Interior half-depth of the box [m].
            wall_half_height: Interior half-height of the side walls [m].
            spacing: Boundary-particle sampling spacing [m].
            floor_y: Height of the box floor [m].
            world_index: World index assigned to all boundary particles.
        """
        if spacing <= 0.0:
            raise ValueError("setup_boundary_particles_box() requires a positive spacing.")

        top_y = floor_y + 2.0 * wall_half_height

        def sample_axis(min_value: float, max_value: float) -> np.ndarray:
            length = max_value - min_value
            count = max(2, int(np.floor(length / spacing + 0.5)) + 1)
            return np.linspace(min_value, max_value, count, dtype=np.float32)

        xs = sample_axis(-half_width, half_width)
        ys = sample_axis(floor_y, top_y)
        zs = sample_axis(-half_depth, half_depth)

        def append_plane(
            points: list[np.ndarray], fixed_axis: int, fixed_value: float, axis_a: np.ndarray, axis_b: np.ndarray
        ) -> None:
            grid_a, grid_b = np.meshgrid(axis_a, axis_b, indexing="ij")
            plane = np.zeros((grid_a.size, 3), dtype=np.float32)
            free_axes = [0, 1, 2]
            free_axes.remove(fixed_axis)
            plane[:, fixed_axis] = fixed_value
            plane[:, free_axes[0]] = grid_a.reshape(-1)
            plane[:, free_axes[1]] = grid_b.reshape(-1)
            points.append(plane)

        planes: list[np.ndarray] = []
        append_plane(planes, 1, floor_y, xs, zs)
        append_plane(planes, 0, half_width, ys, zs)
        append_plane(planes, 0, -half_width, ys, zs)
        append_plane(planes, 2, half_depth, xs, ys)
        append_plane(planes, 2, -half_depth, xs, ys)

        positions = np.concatenate(planes, axis=0)
        quantized = np.round(positions / spacing).astype(np.int64)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        positions = positions[np.sort(unique_indices)]

        self.setup_boundary_particles(
            positions,
            world_indices=np.full(positions.shape[0], world_index, dtype=np.int32),
            spacing=spacing,
        )

    def _has_shape_boundary_contacts(self, contacts: Contacts | None) -> bool:
        """Return whether shape boundary projection can be applied."""
        return contacts is not None and self.model.shape_count > 0 and contacts.soft_contact_max > 0

    def _has_boundary_particles(self) -> bool:
        """Return whether the solver currently owns static boundary particles."""
        return self.boundary_mode == int(self.Config.BoundaryMode.BOUNDARY_PARTICLES) and self._boundary_particle_count > 0

    def _apply_shape_boundary_contacts(
        self,
        state: State,
        particle_q: wp.array(dtype=wp.vec3),
        contacts: Contacts | None,
    ) -> None:
        """Project the given particle positions out of shape contacts."""
        if not self._has_shape_boundary_contacts(contacts):
            return

        model = self.model
        state.particle_q.assign(particle_q)
        model.collide(state, contacts)

        wp.launch(
            project_particle_shape_contacts,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
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

        particle_q.assign(state.particle_q)

    def _apply_shape_boundary_velocity_projection(self, state: State, contacts: Contacts | None) -> None:
        """Project final particle velocities against active particle-shape contacts."""
        if not self._has_shape_boundary_contacts(contacts):
            return

        model = self.model
        model.collide(state, contacts)

        wp.launch(
            initialize_particle_shape_boundary_velocity_projection,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
            ],
            outputs=[
                self._boundary_projected_particle_qd,
                self._boundary_projected_contact_count,
            ],
            device=model.device,
        )

        wp.launch(
            accumulate_particle_shape_boundary_velocity_projection,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_qd,
                model.particle_flags,
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
                model.particle_flags,
                self._boundary_projected_particle_qd,
                self._boundary_projected_contact_count,
            ],
            device=model.device,
        )

    def _apply_xsph_velocity_smoothing(self, state: State) -> None:
        """Apply an optional XSPH-style velocity smoothing pass."""
        if self.xsph_coefficient <= 0.0 or self.model.particle_count == 0:
            return

        model = self.model
        has_boundary_particles = self._has_boundary_particles()
        boundary_grid_id = self._boundary_particle_grid.id if has_boundary_particles and self._boundary_particle_grid else 0

        if model.particle_count > 1 and model.particle_grid is not None:
            wp.launch(
                apply_xsph_velocity_smoothing,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state.particle_q,
                    state.particle_qd,
                    state.ipbf.density,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.rest_density,
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
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    self.xsph_coefficient,
                ],
                outputs=[self._xsph_particle_qd],
                device=model.device,
            )

        state.particle_qd.assign(self._xsph_particle_qd)

    def _apply_viscosity_velocity_diffusion(self, state: State, dt: float) -> None:
        """Apply an optional SPH viscosity diffusion pass to the final velocities."""
        if self.viscosity_coefficient <= 0.0 or self.model.particle_count == 0 or dt <= 0.0:
            return

        model = self.model
        has_boundary_particles = self._has_boundary_particles()
        boundary_grid_id = self._boundary_particle_grid.id if has_boundary_particles and self._boundary_particle_grid else 0

        if model.particle_count > 1 and model.particle_grid is not None:
            wp.launch(
                apply_viscosity_velocity_diffusion,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state.particle_q,
                    state.particle_qd,
                    state.ipbf.density,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.smoothing_radius,
                    self.kernel_family,
                    self.viscosity_coefficient,
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
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.smoothing_radius,
                    self.kernel_family,
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
        """
        model = self.model
        compliance_value = self.compliance if compliance is None else float(compliance)
        has_boundary_particles = self._has_boundary_particles()
        boundary_grid_id = self._boundary_particle_grid.id if has_boundary_particles and self._boundary_particle_grid else 0

        if recompute_density_constraint and model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(particle_q, radius=self.smoothing_radius)

            wp.launch(
                compute_density_and_neighbor_count,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.rest_density,
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
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
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
            wp.launch(
                initialize_density_and_neighbor_count,
                dim=model.particle_count,
                inputs=[
                    particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.rest_density,
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
                    particle_q,
                    state.ipbf.density,
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
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
                    model.particle_flags,
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
                    model.particle_flags,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
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
                    model.particle_flags,
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
                    model.particle_flags,
                    particle_q,
                    model.particle_world,
                    boundary_grid_id,
                    self._boundary_particle_q,
                    self._boundary_particle_volume,
                    self._boundary_particle_world,
                    self._boundary_particle_count,
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
        8. If particle-shape contacts are provided, project iterates back out
           of penetrating static or kinematic boundaries after each update.
        9. Recompute the diagnostic fields on the final positions.
        10. Commit the converged positions and reconstruct velocities, applying
           the artificial damping correction when enabled.

        The current force and Hessian use a stable approximation that includes
        neighborhood Gauss-Newton terms and diagonalized second-order
        stabilization. Particle-shape contacts are handled by a minimal
        normal-direction positional projection using Newton soft contacts.
        """
        if state_in.particle_q is None or state_in.particle_qd is None:
            raise ValueError("SolverIPBF requires particle positions and velocities.")

        if state_in.particle_f is None or state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("SolverIPBF requires particle forces and writable output particle state.")

        if not hasattr(state_out, "ipbf"):
            raise ValueError("State is missing IPBF attributes. Rebuild the model with SolverIPBF.register_custom_attributes().")

        model = self.model
        if model.particle_count == 0:
            return

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
                model.particle_flags,
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

        if self.iterations <= 0:
            self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_guess, contacts)
            state_out.ipbf.x_new.assign(state_out.ipbf.x_guess)
            state_out.ipbf.x_star.assign(state_out.ipbf.x_guess)
            self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)
            state_out.ipbf.delta_q.zero_()
        else:
            for iteration in range(self.iterations):
                is_last_iteration = iteration == self.iterations - 1
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
                            model.particle_flags,
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
                            model.particle_flags,
                            state_out.ipbf.x_guess,
                            state_out.ipbf.delta_q,
                            self.relaxation,
                        ],
                        outputs=[state_out.ipbf.x_star],
                        device=model.device,
                    )
                    self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_star, contacts)

                    self._compute_iteration_fields(
                        state_out,
                        state_out.ipbf.x_guess,
                        dt,
                        recompute_density_constraint=False,
                        compliance=self.compliance,
                    )
                elif is_last_iteration:
                    state_out.ipbf.x_star.assign(state_out.ipbf.x_guess)

                wp.launch(
                    solve_local_system,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_flags,
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
                        model.particle_flags,
                        state_out.ipbf.x_guess,
                        state_out.ipbf.delta_q,
                        self.relaxation,
                    ],
                    outputs=[state_out.ipbf.x_new],
                    device=model.device,
                )
                self._apply_shape_boundary_contacts(state_out, state_out.ipbf.x_new, contacts)

                state_out.ipbf.x_guess.assign(state_out.ipbf.x_new)

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
                    model.particle_flags,
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
                    model.particle_flags,
                    dt,
                ],
                outputs=[state_out.particle_qd],
                device=model.device,
            )

        self._apply_shape_boundary_velocity_projection(state_out, contacts)
        self._apply_viscosity_velocity_diffusion(state_out, dt)
        self._apply_xsph_velocity_smoothing(state_out)
