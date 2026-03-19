# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverIPBF
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_ipbf_registers_attributes_and_applies_inertial_prediction(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 1.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=2.0,
        radius=0.1,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    config = SolverIPBF.Config(
        rest_density=1234.0,
        smoothing_radius=0.25,
        compliance=0.125,
    )
    solver = SolverIPBF(model, config)

    state_0 = model.state()
    state_1 = model.state()

    dt = 0.1
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    test.assertTrue(hasattr(model, "ipbf"))
    test.assertTrue(hasattr(state_1, "ipbf"))

    test.assertAlmostEqual(float(model.ipbf.rest_density.numpy()[0]), config.rest_density, places=6)
    test.assertAlmostEqual(float(model.ipbf.smoothing_radius.numpy()[0]), config.smoothing_radius, places=6)
    test.assertAlmostEqual(float(model.ipbf.compliance.numpy()[0]), config.compliance, places=6)

    expected_y = np.array([0.0, 1.0 - 9.81 * dt * dt, 0.0], dtype=np.float32)
    expected_v = np.array([0.0, -9.81 * dt, 0.0], dtype=np.float32)

    np.testing.assert_allclose(state_1.ipbf.y.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.density.numpy(), np.zeros(1, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_ipbf_requires_registered_attributes(test, device):
    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=1.0,
        radius=0.1,
    )
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "requires IPBF custom attributes"):
        SolverIPBF(model)


devices = get_test_devices(mode="basic")


class TestSolverIPBF(unittest.TestCase):
    pass


add_function_test(
    TestSolverIPBF,
    "test_ipbf_registers_attributes_and_applies_inertial_prediction",
    test_ipbf_registers_attributes_and_applies_inertial_prediction,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_requires_registered_attributes",
    test_ipbf_requires_registered_attributes,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
