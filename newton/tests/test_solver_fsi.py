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
            fsi_projection_reaction_relaxation=1.0,
            fsi_velocity_projection_reaction_relaxation=1.0,
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
            fsi_projection_reaction_relaxation=1.0,
            fsi_velocity_projection_reaction_relaxation=1.0,
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


def build_single_particle_cloth_fsi(device):
    """Build a one-fluid-particle / one-cloth-triangle coupling scene."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
    builder.color()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def run_loose_cloth_fsi_step(device):
    model = build_single_particle_cloth_fsi(device)

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    fluid_solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_triangle_contact_enabled=True,
            fsi_triangle_contact_margin=0.0,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )
    solid_solver = SolverVBD(
        model,
        iterations=1,
        particle_start=1,
        particle_count=3,
        fsi_boundary_model=boundary_model,
    )
    solver = SolverFSI(model, fluid_solver, solid_solver, boundary_model)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.05
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    return state_0, state_1, solver, boundary_model, dt


def run_interlinked_cloth_fsi_step(device):
    model = build_single_particle_cloth_fsi(device)

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    fluid_solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=1,
            relaxation=1.0,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_triangle_contact_enabled=True,
            fsi_triangle_contact_margin=0.0,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )
    solid_solver = SolverVBD(
        model,
        iterations=1,
        particle_start=1,
        particle_count=3,
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

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.05
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    return state_0, state_1, solver, boundary_model, dt


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


def test_solver_fsi_loose_coupling_moves_cloth_from_ipbf_triangle_contact(test, device):
    state_0, state_1, solver, boundary_model, dt = run_loose_cloth_fsi_step(device)

    initial_q = state_0.particle_q.numpy()
    final_q = state_1.particle_q.numpy()
    final_qd = state_1.particle_qd.numpy()
    vertex_contact_delta = boundary_model.vertex_contact_delta.numpy()
    vertex_force = boundary_model.vertex_force.numpy()
    fluid_delta = solver.fluid_solver._triangle_contact_particle_delta.numpy()[0]

    test.assertGreater(float(fluid_delta[2]), 0.0)
    test.assertTrue(np.all(vertex_contact_delta[1:, 2] < 0.0))
    np.testing.assert_allclose(final_q[0], solver._fluid_state.particle_q.numpy()[0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(final_q[1:] - initial_q[1:], vertex_contact_delta[1:], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(final_qd[1:], vertex_contact_delta[1:] / dt, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[1:] * (dt * dt), vertex_contact_delta[1:], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[0], np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)


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


def test_solver_fsi_interlinked_coupling_preserves_fluid_subset_state_for_cloth(test, device):
    state_0, state_1, solver, boundary_model, _dt = run_interlinked_cloth_fsi_step(device)

    initial_q = state_0.particle_q.numpy()
    final_q = state_1.particle_q.numpy()
    fluid_q = solver._fluid_state.particle_q.numpy()
    vertex_force = boundary_model.vertex_force.numpy()

    test.assertGreater(float(final_q[0, 2] - initial_q[0, 2]), 0.0)
    np.testing.assert_allclose(final_q[0], fluid_q[0], rtol=1.0e-6, atol=1.0e-6)
    test.assertGreaterEqual(int(solver.fluid_solver._triangle_contact_pair_count.numpy()[0]), 1)
    test.assertTrue(np.all(vertex_force[1:, 2] < 0.0))
    test.assertTrue(np.all(final_q[1:, 2] <= initial_q[1:, 2]))


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
    "test_solver_fsi_loose_coupling_moves_cloth_from_ipbf_triangle_contact",
    test_solver_fsi_loose_coupling_moves_cloth_from_ipbf_triangle_contact,
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
add_function_test(
    TestSolverFSI,
    "test_solver_fsi_interlinked_coupling_preserves_fluid_subset_state_for_cloth",
    test_solver_fsi_interlinked_coupling_preserves_fluid_subset_state_for_cloth,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
