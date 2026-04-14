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

"""Tests for AVBD/IPBF force handoff into SolverVBD."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import FSIBoundaryModel, SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


def run_vbd_fsi_body_force_step(device, *, fsi_force_relaxation: float = 1.0):
    """Run one VBD step with a prescribed FSI body force."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        mass=1.0,
        lock_inertia=True,
    )
    builder.add_shape_box(body=body, hx=0.25, hy=0.25, hz=0.25)
    builder.color()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(model, spacing=0.25, support_radius=0.3, device=device)
    body_force = boundary_model.body_force.numpy()
    body_force[body] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    boundary_model.body_force.assign(body_force)

    solver = SolverVBD(
        model,
        iterations=1,
        fsi_boundary_model=boundary_model,
        fsi_force_relaxation=fsi_force_relaxation,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    initial_q = state_0.body_q.numpy()[body].copy()

    dt = 0.1
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    return initial_q, state_1, solver, boundary_model, body, dt


def test_vbd_fsi_body_force_moves_rigid_body(test, device):
    initial_q, state_1, solver, boundary_model, body, dt = run_vbd_fsi_body_force_step(device)

    mass = float(solver.model.body_mass.numpy()[body])
    force = boundary_model.body_force.numpy()[body]
    final_q = state_1.body_q.numpy()[body]
    final_qd = state_1.body_qd.numpy()[body]
    expected_delta = force * (dt * dt) / mass
    expected_velocity = force * dt / mass

    np.testing.assert_allclose(final_q[:3] - initial_q[:3], expected_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(final_qd[:3], expected_velocity, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(solver.body_forces.numpy()[body], force, rtol=1.0e-6, atol=1.0e-6)


def test_vbd_fsi_force_relaxation_scales_body_response(test, device):
    initial_q, state_1, solver, boundary_model, body, dt = run_vbd_fsi_body_force_step(
        device, fsi_force_relaxation=0.25
    )

    mass = float(solver.model.body_mass.numpy()[body])
    force = boundary_model.body_force.numpy()[body] * 0.25
    final_q = state_1.body_q.numpy()[body]
    expected_delta = force * (dt * dt) / mass

    np.testing.assert_allclose(final_q[:3] - initial_q[:3], expected_delta, rtol=1.0e-5, atol=1.0e-6)


def test_vbd_reset_restores_rigid_history(test, device):
    initial_q, _state_1, solver, _, body, _ = run_vbd_fsi_body_force_step(device)

    moved_history = solver.body_q_prev.numpy()[body]
    test.assertGreater(float(moved_history[0] - initial_q[0]), 0.0)

    reset_state = solver.model.state()
    solver.reset(reset_state)
    np.testing.assert_allclose(
        solver.body_q_prev.numpy()[body], reset_state.body_q.numpy()[body], rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(solver.body_q_prev.numpy()[body], initial_q, rtol=1.0e-6, atol=1.0e-6)


devices = get_test_devices()


class TestSolverVBDFSI(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDFSI,
    "test_vbd_fsi_body_force_moves_rigid_body",
    test_vbd_fsi_body_force_moves_rigid_body,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverVBDFSI,
    "test_vbd_fsi_force_relaxation_scales_body_response",
    test_vbd_fsi_force_relaxation_scales_body_response,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverVBDFSI,
    "test_vbd_reset_restores_rigid_history",
    test_vbd_reset_restores_rigid_history,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
