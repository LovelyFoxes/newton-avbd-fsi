"""Implicit Position-Based Fluids solver."""

from __future__ import annotations

from dataclasses import dataclass

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
    initialize_guess_positions,
    predict_inertial_positions,
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
            compliance: Normalized compliance parameter ``alpha = 1 / k``.
            iterations: Number of relaxed Jacobi iterations.
            relaxation: Position update relaxation factor. The paper uses 0.5.
            use_constraint_clamp: Clamp negative density constraints at free
                surfaces to zero.
            damping_compliance: Compliance used for the alternative position
                solve in artificial damping.
            damping_beta: Distance threshold factor used by the damping model.
        """

        rest_density: float = 1000.0
        smoothing_radius: float = 0.1
        compliance: float = 0.0
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
        self.compliance = float(self.config.compliance)
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

        TODO:
            1. Compute inertial target positions ``y``.
            2. Initialize ``x_guess <- y``.
            3. Build/update the particle hash grid from ``x_guess``.
            4. Run relaxed Jacobi iterations to compute densities, Hessians, and
               local updates ``delta_q``.
            5. Commit ``x_new`` to ``state_out.particle_q``.
            6. Reconstruct velocities from position changes and apply optional
               artificial damping.

        The current implementation is intentionally minimal and only executes
        the shell of an IPBF step:

        1. Predict inertial target positions ``y``.
        2. Initialize ``x_guess`` and ``x_new`` from ``y``.
        3. Commit ``x_new`` to ``state_out.particle_q``.
        4. Reconstruct ``state_out.particle_qd`` from position changes.

        Pressure iterations, density evaluation, Hessian assembly, and relaxed
        Jacobi updates will be added on top of this scaffold.
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

        state_out.particle_q.assign(state_out.ipbf.x_new)
        wp.launch(
            update_velocity_from_positions,
            dim=model.particle_count,
            inputs=[
                state_out.ipbf.x_new,
                state_in.particle_q,
                model.particle_flags,
                dt,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )

        state_out.ipbf.density.zero_()
        state_out.ipbf.delta_q.zero_()
