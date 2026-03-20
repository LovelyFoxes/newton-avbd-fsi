import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverIPBF
from newton.tests.unittest_utils import add_function_test, get_test_devices


def poly6_density(mass: float, support_radius: float, distance: float) -> float:
    """Reference 3D poly6 density contribution used by the IPBF kernels."""
    if support_radius <= 0.0 or distance >= support_radius:
        return 0.0

    support_radius2 = support_radius * support_radius
    x = support_radius2 - distance * distance
    return mass * 315.0 / (64.0 * np.pi * support_radius**9) * x**3


def poly6_gradient_contribution(mass: float, support_radius: float, displacement: np.ndarray) -> np.ndarray:
    """Reference 3D poly6 gradient contribution used by the IPBF kernels."""
    distance2 = float(np.dot(displacement, displacement))
    if support_radius <= 0.0 or distance2 >= support_radius * support_radius or distance2 == 0.0:
        return np.zeros(3, dtype=np.float32)

    x = support_radius * support_radius - distance2
    scale = -945.0 / (32.0 * np.pi * support_radius**9) * x * x
    return (mass * scale * displacement).astype(np.float32)


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
    expected_density = poly6_density(2.0, config.smoothing_radius, 0.0)

    np.testing.assert_allclose(state_1.ipbf.y.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.density.numpy(), np.array([expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(state_1.ipbf.constraint.numpy(), np.zeros(1, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.constraint_gradient.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
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


def test_ipbf_multi_particle_shell_step_builds_neighbor_search(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.2,
            iterations=2,
        ),
    )

    test.assertIsNotNone(model.particle_grid)

    state_0 = model.state()
    state_1 = model.state()

    dt = 0.05
    for _ in range(3):
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    q = state_0.particle_q.numpy()
    qd = state_0.particle_qd.numpy()
    density = state_0.ipbf.density.numpy()
    neighbor_count = state_0.ipbf.neighbor_count.numpy()

    test.assertEqual(q.shape, (2, 3))
    test.assertEqual(qd.shape, (2, 3))
    test.assertLess(q[0, 1], 1.0)
    test.assertLess(q[1, 1], 1.0)
    test.assertAlmostEqual(q[0, 0], -0.05, places=5)
    test.assertAlmostEqual(q[1, 0], 0.05, places=5)
    test.assertAlmostEqual(q[0, 1], q[1, 1], places=5)
    test.assertAlmostEqual(qd[0, 1], qd[1, 1], places=5)
    np.testing.assert_array_equal(neighbor_count, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(density[0], density[1], rtol=1e-5, atol=1e-5)
    test.assertGreater(float(density[0]), 0.0)
    np.testing.assert_allclose(state_0.ipbf.constraint.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_0.ipbf.constraint_gradient.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_ipbf_computes_constraint_and_gradient(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    rest_density = 1.0
    support_radius = 0.2
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            use_constraint_clamp=False,
        ),
    )

    state_0 = model.state()
    state_1 = model.state()

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    density = state_1.ipbf.density.numpy()
    constraint = state_1.ipbf.constraint.numpy()
    constraint_gradient = state_1.ipbf.constraint_gradient.numpy()

    distance = 0.1
    expected_density = poly6_density(1.0, support_radius, 0.0) + poly6_density(1.0, support_radius, distance)
    expected_constraint = expected_density / rest_density - 1.0
    expected_gradient_0 = poly6_gradient_contribution(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_gradient_1 = -expected_gradient_0

    np.testing.assert_allclose(density, np.array([expected_density, expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        constraint,
        np.array([expected_constraint, expected_constraint], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(constraint_gradient[0], expected_gradient_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(constraint_gradient[1], expected_gradient_1, rtol=1e-5, atol=1e-5)
    test.assertGreater(float(constraint[0]), 0.0)


def test_ipbf_reset_restores_initial_particle_state(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.1, 0.0),
        ],
        vel=[
            wp.vec3(0.2, 0.0, 0.0),
            wp.vec3(-0.2, 0.0, 0.0),
        ],
        mass=[1.0, 1.5],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))
    solver = SolverIPBF(model, SolverIPBF.Config(smoothing_radius=0.2))

    state_0 = model.state()
    state_1 = model.state()

    q_initial = state_0.particle_q.numpy().copy()
    qd_initial = state_0.particle_qd.numpy().copy()

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    q_advanced = state_1.particle_q.numpy()
    test.assertFalse(np.allclose(q_advanced, q_initial))

    solver.reset(state_1)

    np.testing.assert_allclose(state_1.particle_q.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.particle_qd.numpy(), qd_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.particle_f.numpy(), np.zeros_like(q_initial), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.y.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.density.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(state_1.ipbf.constraint.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.constraint_gradient.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


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

add_function_test(
    TestSolverIPBF,
    "test_ipbf_multi_particle_shell_step_builds_neighbor_search",
    test_ipbf_multi_particle_shell_step_builds_neighbor_search,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_computes_constraint_and_gradient",
    test_ipbf_computes_constraint_and_gradient,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_reset_restores_initial_particle_state",
    test_ipbf_reset_restores_initial_particle_state,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
