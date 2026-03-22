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
    apply_relaxed_jacobi_update,
    compute_constraint_and_gradient,
    compute_density_and_neighbor_count,
    compute_force,
    compute_force_without_grid,
    compute_hessian,
    compute_hessian_without_grid,
    initialize_guess_positions,
    initialize_constraint_and_gradient,
    initialize_density_and_neighbor_count,
    predict_inertial_positions,
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
        self.compliance = float(self.config.compliance)
        self.hessian_regularization = float(self.config.hessian_regularization)
        self.iterations = int(self.config.iterations)
        self.relaxation = float(self.config.relaxation)
        self.use_constraint_clamp = bool(self.config.use_constraint_clamp)
        self.damping_compliance = float(self.config.damping_compliance)
        self.damping_beta = float(self.config.damping_beta)

        if not hasattr(model, "ipbf"):
            raise ValueError(
                "SolverIPBF requires IPBF custom attributes. "
                "Call SolverIPBF.register_custom_attributes(builder) before finalize()."
            )

        model.ipbf.rest_density.fill_(self.rest_density)
        model.ipbf.smoothing_radius.fill_(self.smoothing_radius)
        model.ipbf.compliance.fill_(self.compliance)

        self._initial_state = model.state()

        if model.particle_count > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

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

        state_out.ipbf.density.zero_()
        state_out.ipbf.neighbor_count.zero_()
        state_out.ipbf.constraint.zero_()
        state_out.ipbf.constraint_gradient.zero_()
        state_out.ipbf.force.zero_()
        state_out.ipbf.delta_q.zero_()

        if self.model.particle_count > 1 and self.model.particle_grid is not None:
            with wp.ScopedDevice(self.model.device):
                self.model.particle_grid.build(state_out.ipbf.x_guess, radius=self.smoothing_radius)

    def _compute_iteration_fields(
        self,
        state: State,
        particle_q: wp.array(dtype=wp.vec3),
        dt: float,
    ) -> None:
        """Assemble IPBF local solve quantities for the given particle positions.

        Args:
            state: State whose ``ipbf`` buffers receive the assembled quantities.
            particle_q: Current particle positions [m] used to build the local
                IPBF system.
            dt: Timestep size [s].
        """
        model = self.model

        if model.particle_count > 1 and model.particle_grid is not None:
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
                    state.ipbf.density,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    int(self.use_constraint_clamp),
                ],
                outputs=[state.ipbf.constraint, state.ipbf.constraint_gradient],
                device=model.device,
            )
        else:
            wp.launch(
                initialize_density_and_neighbor_count,
                dim=model.particle_count,
                inputs=[
                    model.particle_mass,
                    model.particle_flags,
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
                    model.particle_flags,
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
                    model.particle_flags,
                    model.particle_world,
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    self.compliance,
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
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    self.compliance,
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
                    self.compliance,
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
                    state.ipbf.constraint,
                    state.ipbf.constraint_gradient,
                    self.rest_density,
                    self.smoothing_radius,
                    self.kernel_family,
                    self.compliance,
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
        7. Recompute the diagnostic fields on the final positions.
        8. Commit the converged positions and reconstruct velocities.

        The current force and Hessian use a first-pass stable approximation.
        The full paper-specific Hessian refinement and artificial damping are
        still to be added.
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

        if self.iterations <= 0:
            self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)
            state_out.ipbf.delta_q.zero_()
        else:
            for _ in range(self.iterations):
                self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)

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

                state_out.ipbf.x_guess.assign(state_out.ipbf.x_new)

            self._compute_iteration_fields(state_out, state_out.ipbf.x_guess, dt)

        state_out.particle_q.assign(state_out.ipbf.x_guess)
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
