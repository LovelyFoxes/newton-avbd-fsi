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

"""Tests for loose AVBD/IPBF fluid-solid coupling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


def build_single_particle_box_fsi(device):
    """Build a small IPBF particle / AVBD box coupling scene."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 0.12, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=2.0,
        radius=0.05,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        lock_inertia=True,
    )
    builder.add_shape_box(
        body=body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.25,
        hy=0.1,
        hz=0.25,
    )
    builder.color()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model, body


def run_loose_fsi_step(device):
    model, body = build_single_particle_box_fsi(device)

    boundary_model = FSIBoundaryModel(model, spacing=0.25, support_radius=0.3, device=device)
    fluid_solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.3,
            iterations=0,
            fsi_reaction_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )
    solid_solver = SolverVBD(
        model,
        iterations=1,
        integrate_particles=False,
        fsi_boundary_model=boundary_model,
    )
    solver = SolverFSI(model, fluid_solver, solid_solver, boundary_model)

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.1
    solver.step(state_0, state_1, control=None, contacts=contacts, dt=dt)

    return state_0, state_1, solver, boundary_model, contacts, body, dt


def run_interlinked_fsi_step(device):
    model, body = build_single_particle_box_fsi(device)

    boundary_model = FSIBoundaryModel(model, spacing=0.25, support_radius=0.3, device=device)
    fluid_solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.3,
            iterations=0,
            fsi_reaction_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )
    solid_solver = SolverVBD(
        model,
        iterations=1,
        integrate_particles=False,
        fsi_boundary_model=boundary_model,
    )
    solver = SolverFSI(
        model,
        fluid_solver,
        solid_solver,
        boundary_model,
        SolverFSI.Config(
            mode=SolverFSI.Config.CouplingMode.INTERLINKED,
            coupling_iterations=1,
        ),
    )

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.1
    solver.step(state_0, state_1, control=None, contacts=contacts, dt=dt)

    return state_0, state_1, solver, boundary_model, contacts, body, dt


def test_solver_fsi_loose_coupling_moves_body_from_ipbf_reaction(test, device):
    state_0, state_1, solver, boundary_model, contacts, body, dt = run_loose_fsi_step(device)

    particle_delta = solver.fluid_solver._boundary_projection_delta_total.numpy()[0]
    body_force = boundary_model.body_force.numpy()[body]
    body_mass = float(solver.model.body_mass.numpy()[body])
    body_delta = state_1.body_q.numpy()[body, :3] - state_0.body_q.numpy()[body, :3]
    expected_body_delta = body_force * (dt * dt) / body_mass

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(float(particle_delta[1]), 0.0)
    test.assertLess(float(body_force[1]), 0.0)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0, 1], 0.15, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(body_delta, expected_body_delta, rtol=1.0e-5, atol=1.0e-6)
    test.assertGreater(float(state_1.ipbf.density.numpy()[0]), 0.0)


def test_solver_fsi_interlinked_coupling_moves_body_from_ipbf_reaction(test, device):
    state_0, state_1, solver, boundary_model, contacts, body, dt = run_interlinked_fsi_step(device)

    particle_delta = solver.fluid_solver._boundary_projection_delta_total.numpy()[0]
    body_force = boundary_model.body_force.numpy()[body]
    body_mass = float(solver.model.body_mass.numpy()[body])
    body_delta = state_1.body_q.numpy()[body, :3] - state_0.body_q.numpy()[body, :3]
    expected_body_delta = body_force * (dt * dt) / body_mass

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(float(particle_delta[1]), 0.0)
    test.assertLess(float(body_force[1]), 0.0)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0, 1], 0.15, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(body_delta, expected_body_delta, rtol=1.0e-5, atol=1.0e-6)
    test.assertGreater(float(state_1.ipbf.density.numpy()[0]), 0.0)


devices = get_test_devices()


class TestSolverFSI(unittest.TestCase):
    pass


add_function_test(
    TestSolverFSI,
    "test_solver_fsi_loose_coupling_moves_body_from_ipbf_reaction",
    test_solver_fsi_loose_coupling_moves_body_from_ipbf_reaction,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverFSI,
    "test_solver_fsi_interlinked_coupling_moves_body_from_ipbf_reaction",
    test_solver_fsi_interlinked_coupling_moves_body_from_ipbf_reaction,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
