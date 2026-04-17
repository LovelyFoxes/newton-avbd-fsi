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

"""Fluid-solid coupling wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .boundary_model import FSIBoundaryModel

__all__ = ["SolverFSI"]


class SolverFSI(SolverBase):
    """Fluid-solid solver wrapper.

    `SolverFSI` owns a fluid solver, a solid solver, and an
    :class:`FSIBoundaryModel`. The default loose schedule advances the fluid
    solver first, then lets the solid solver consume the accumulated FSI body
    wrenches. The interlinked schedule alternates split IPBF and AVBD
    iterations inside one timestep so updated rigid boundary samples can feed
    back into the next fluid iteration.

    Args:
        model: Newton model shared by all participating solvers.
        fluid_solver: Solver responsible for fluid particles.
        solid_solver: Solver responsible for solid bodies.
        boundary_model: Boundary model used as the coupling data interface.
        config: Optional coupling configuration.

    Returns:
        Fluid-solid coupling solver.
    """

    @dataclass
    class Config:
        """Fluid-solid coupling wrapper configuration.

        Attributes:
            mode: Coupling schedule.
            coupling_iterations: Number of interlinked IPBF/AVBD iteration
                pairs. If ``None``, uses the larger ``iterations`` value found
                on the participating solvers, clamped to at least one.
            pass_contacts_to_solid: Whether the same contact buffer passed to
                :meth:`step` should also be passed to the solid solver.
            update_boundary_after_solid: Whether to refresh boundary sample
                world positions [m], velocities [m/s], and grid after the solid
                solver advances.
            copy_fluid_particle_state: Whether to copy fluid particle state and
                fluid diagnostic namespaces back after the solid solver runs. If
                ``None``, this is enabled when the solid solver exposes
                ``integrate_particles=False``.
        """

        class CouplingMode(IntEnum):
            """Supported FSI coupling schedules."""

            LOOSE = 0
            INTERLINKED = 1

        mode: CouplingMode = CouplingMode.LOOSE
        coupling_iterations: int | None = None
        pass_contacts_to_solid: bool = True
        update_boundary_after_solid: bool = True
        copy_fluid_particle_state: bool | None = None

    def __init__(
        self,
        model: Model,
        fluid_solver: SolverBase,
        solid_solver: SolverBase,
        boundary_model: FSIBoundaryModel,
        config: Config | None = None,
    ):
        super().__init__(model)

        self.config = config if config is not None else self.Config()
        self.fluid_solver = fluid_solver
        self.solid_solver = solid_solver
        self.boundary_model = boundary_model
        self._fluid_state = model.state()

        self._validate_solver_model("fluid_solver", fluid_solver)
        self._validate_solver_model("solid_solver", solid_solver)
        self._validate_boundary_model(boundary_model)
        self._configure_participant_solvers()

    def _validate_solver_model(self, name: str, solver: SolverBase) -> None:
        if getattr(solver, "model", self.model) is not self.model:
            raise ValueError(f"SolverFSI {name} must be built from this solver's model.")

    def _validate_boundary_model(self, boundary_model: FSIBoundaryModel) -> None:
        if boundary_model.model is not self.model:
            raise ValueError("SolverFSI boundary model must be built from this solver's model.")
        if boundary_model.device != self.model.device:
            raise ValueError("SolverFSI boundary model device must match the solver model device.")

    def _configure_participant_solvers(self) -> None:
        if hasattr(self.fluid_solver, "set_boundary_model"):
            self.fluid_solver.set_boundary_model(self.boundary_model)

        if hasattr(self.solid_solver, "set_fsi_boundary_model"):
            self.solid_solver.set_fsi_boundary_model(self.boundary_model)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance a loose-coupled FSI step.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Optional control inputs passed through to both solvers.
            contacts: Contact data passed to the fluid solver, and optionally
                to the solid solver depending on :attr:`Config.pass_contacts_to_solid`.
            dt: Time step size [s].
        """
        self.boundary_model.clear_step_diagnostics()

        if self.config.mode == self.Config.CouplingMode.LOOSE:
            self._step_loose(state_in, state_out, control, contacts, dt)
        elif self.config.mode == self.Config.CouplingMode.INTERLINKED:
            self._step_interlinked(state_in, state_out, control, contacts, dt)
        else:
            raise ValueError(f"Unsupported SolverFSI coupling mode: {self.config.mode}")

    def _step_loose(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one loose-coupled FSI timestep."""
        self.fluid_solver.step(state_in, self._fluid_state, control, contacts, dt)
        self.boundary_model.accumulate_step_diagnostics()

        solid_contacts = contacts if self.config.pass_contacts_to_solid else None
        self.solid_solver.step(self._fluid_state, state_out, control, solid_contacts, dt)

        if self._should_copy_fluid_particle_state():
            self._copy_particle_state(self._fluid_state, state_out)
            self._copy_custom_state_namespaces(self._fluid_state, state_out)

        if self.config.update_boundary_after_solid:
            self.boundary_model.update_world_kinematics(state_out)
            self.boundary_model.build_grid()

    def _step_interlinked(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one timestep with alternating fluid and solid iterations."""
        self._require_interlinked_hooks()

        solid_contacts = contacts if self.config.pass_contacts_to_solid else None
        fluid_active = self.fluid_solver._begin_step(state_in, self._fluid_state, dt)
        if not fluid_active:
            self.solid_solver.step(state_in, state_out, control, solid_contacts, dt)
            if self.config.update_boundary_after_solid:
                self._refresh_boundary(state_out)
            return

        self.solid_solver._begin_step(self._fluid_state, state_out, solid_contacts, dt)

        coupling_iterations = self._coupling_iteration_count()
        fluid_iterations = int(getattr(self.fluid_solver, "iterations", coupling_iterations))
        for iteration in range(coupling_iterations):
            self.boundary_model.clear_forces()
            if fluid_iterations <= 0:
                if iteration == 0:
                    self.fluid_solver._solve_zero_iteration(self._fluid_state, contacts, dt)
            else:
                self.fluid_solver._solve_iteration(
                    self._fluid_state,
                    contacts,
                    dt,
                    iteration,
                    coupling_iterations,
                )
            self.boundary_model.accumulate_step_diagnostics()
            self.solid_solver._solve_iteration(
                self._fluid_state,
                state_out,
                solid_contacts,
                dt,
                iteration,
            )
            self._copy_body_state(state_out, self._fluid_state)
            self._refresh_boundary(state_out)

        self.fluid_solver._finalize_step(state_in, self._fluid_state, contacts, dt)
        self.solid_solver._finalize_step(self._fluid_state, state_out, dt)

        if self._should_copy_fluid_particle_state():
            self._copy_particle_state(self._fluid_state, state_out)
            self._copy_custom_state_namespaces(self._fluid_state, state_out)

        if self.config.update_boundary_after_solid:
            self._refresh_boundary(state_out)

    def _require_interlinked_hooks(self) -> None:
        """Ensure the participating solvers expose split-step hooks."""
        for name, solver in (("fluid_solver", self.fluid_solver), ("solid_solver", self.solid_solver)):
            for method_name in ("_begin_step", "_solve_iteration", "_finalize_step"):
                if not hasattr(solver, method_name):
                    raise TypeError(f"SolverFSI INTERLINKED mode requires {name} to provide {method_name}().")

        if int(getattr(self.fluid_solver, "iterations", 1)) <= 0 and not hasattr(
            self.fluid_solver, "_solve_zero_iteration"
        ):
            raise TypeError(
                "SolverFSI INTERLINKED mode requires fluid_solver to provide _solve_zero_iteration() "
                "when fluid iterations are zero."
            )

    def _coupling_iteration_count(self) -> int:
        """Return the number of interlinked iteration pairs to execute."""
        if self.config.coupling_iterations is not None:
            coupling_iterations = int(self.config.coupling_iterations)
            if coupling_iterations <= 0:
                raise ValueError("SolverFSI coupling_iterations must be positive when provided.")
            return coupling_iterations

        fluid_iterations = int(getattr(self.fluid_solver, "iterations", 1))
        solid_iterations = int(getattr(self.solid_solver, "iterations", 1))
        return max(fluid_iterations, solid_iterations, 1)

    def _refresh_boundary(self, state: State) -> None:
        """Update boundary sample world kinematics and spatial grid."""
        self.boundary_model.update_world_kinematics(state)
        self.boundary_model.build_grid()

    def _should_copy_fluid_particle_state(self) -> bool:
        if self.config.copy_fluid_particle_state is not None:
            return bool(self.config.copy_fluid_particle_state)

        return not bool(getattr(self.solid_solver, "integrate_particles", True))

    def _copy_particle_state(self, state_src: State, state_dst: State) -> None:
        if self.model.particle_count == 0:
            return

        if state_src.particle_q is not None and state_dst.particle_q is not None:
            state_dst.particle_q.assign(state_src.particle_q)
        if state_src.particle_qd is not None and state_dst.particle_qd is not None:
            state_dst.particle_qd.assign(state_src.particle_qd)
        if state_src.particle_f is not None and state_dst.particle_f is not None:
            state_dst.particle_f.assign(state_src.particle_f)

    def _copy_body_state(self, state_src: State, state_dst: State) -> None:
        if self.model.body_count == 0:
            return

        if state_src.body_q is not None and state_dst.body_q is not None:
            state_dst.body_q.assign(state_src.body_q)
        if state_src.body_qd is not None and state_dst.body_qd is not None:
            state_dst.body_qd.assign(state_src.body_qd)
        if state_src.body_f is not None and state_dst.body_f is not None:
            state_dst.body_f.assign(state_src.body_f)

    def _copy_custom_state_namespaces(self, state_src: State, state_dst: State) -> None:
        for name, namespace_src in state_src.__dict__.items():
            if isinstance(namespace_src, wp.array) or namespace_src is None:
                continue

            namespace_dst = getattr(state_dst, name, None)
            if namespace_dst is None:
                continue

            for attr, value_src in namespace_src.__dict__.items():
                value_dst = getattr(namespace_dst, attr, None)
                if isinstance(value_src, wp.array) and isinstance(value_dst, wp.array):
                    value_dst.assign(value_src)
