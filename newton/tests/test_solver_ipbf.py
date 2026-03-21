import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipbf.ipbf_kernels import kernel_gradient, kernel_hessian, kernel_value
from newton.solvers import SolverIPBF
from newton.tests.unittest_utils import add_function_test, get_test_devices


def kernel_density_contribution(mass: float, support_radius: float, distance: float) -> float:
    """Reference scalar kernel density contribution used by the IPBF kernels."""
    if support_radius <= 0.0 or distance >= support_radius:
        return 0.0

    support_radius2 = support_radius * support_radius
    x = support_radius2 - distance * distance
    return mass * 315.0 / (64.0 * np.pi * support_radius**9) * x**3


def kernel_gradient_contribution(mass: float, support_radius: float, displacement: np.ndarray) -> np.ndarray:
    """Reference spatial kernel gradient contribution used by the IPBF kernels."""
    distance2 = float(np.dot(displacement, displacement))
    if support_radius <= 0.0 or distance2 >= support_radius * support_radius or distance2 == 0.0:
        return np.zeros(3, dtype=np.float32)

    x = support_radius * support_radius - distance2
    scale = -945.0 / (32.0 * np.pi * support_radius**9) * x * x
    return (mass * scale * displacement).astype(np.float32)


def kernel_hessian_reference(support_radius: float, displacement: np.ndarray) -> np.ndarray:
    """Reference spatial kernel Hessian used by the IPBF kernels."""
    distance2 = float(np.dot(displacement, displacement))
    if support_radius <= 0.0 or distance2 >= support_radius * support_radius:
        return np.zeros((3, 3), dtype=np.float32)

    x = support_radius * support_radius - distance2
    identity = np.eye(3, dtype=np.float32)
    outer = np.outer(displacement, displacement)
    return (
        -945.0 / (32.0 * np.pi * support_radius**9) * x * x * identity
        + 945.0 / (8.0 * np.pi * support_radius**9) * x * outer
    ).astype(np.float32)


def expected_constraint_hessian(
    masses: list[float],
    support_radius: float,
    displacements: list[np.ndarray],
    rest_density: float,
) -> np.ndarray:
    """Reference constraint Hessian built from kernel Hessian contributions."""
    if rest_density <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)

    hessian = np.zeros((3, 3), dtype=np.float32)
    for mass, displacement in zip(masses, displacements, strict=True):
        hessian += mass * kernel_hessian_reference(support_radius, displacement)

    return (hessian / rest_density).astype(np.float32)


def expected_neighbor_constraint_hessian(
    mass_self: float,
    support_radius: float,
    displacement_neighbor_minus_self: np.ndarray,
    rest_density: float,
) -> np.ndarray:
    """Reference neighbor constraint Hessian with respect to the current particle."""
    if rest_density <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)

    return (mass_self * kernel_hessian_reference(support_radius, displacement_neighbor_minus_self) / rest_density).astype(
        np.float32
    )


def expected_ipbf_hessian(
    gradient: np.ndarray,
    mass: float,
    compliance: float,
    dt: float,
    regularization: float,
    constraint: float = 0.0,
    constraint_hessian: np.ndarray | None = None,
    neighbor_gradients: list[np.ndarray] | None = None,
    neighbor_constraints: list[float] | None = None,
    neighbor_constraint_hessians: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Reference IPBF Hessian used by the current skeleton."""
    inertia_scale = 0.0 if dt <= 0.0 else compliance * mass / (dt * dt)
    hessian = ((inertia_scale + regularization) * np.eye(3, dtype=np.float32) + np.outer(gradient, gradient)).astype(
        np.float32
    )

    if neighbor_gradients is not None:
        for neighbor_gradient in neighbor_gradients:
            hessian += np.outer(neighbor_gradient, neighbor_gradient).astype(np.float32)

    if constraint_hessian is not None and constraint != 0.0:
        hessian += abs(constraint) * np.diag(np.linalg.norm(constraint_hessian, axis=0).astype(np.float32))

    if neighbor_constraints is not None and neighbor_constraint_hessians is not None:
        for neighbor_constraint, neighbor_constraint_hessian in zip(
            neighbor_constraints, neighbor_constraint_hessians, strict=True
        ):
            if neighbor_constraint != 0.0:
                hessian += abs(neighbor_constraint) * np.diag(
                    np.linalg.norm(neighbor_constraint_hessian, axis=0).astype(np.float32)
                )

    return hessian.astype(np.float32)


def expected_ipbf_delta(force: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Reference local IPBF position update used by the current skeleton."""
    if abs(float(np.linalg.det(hessian))) <= 1.0e-8:
        return np.zeros(3, dtype=np.float32)

    return np.linalg.solve(hessian, force).astype(np.float32)


def expected_two_particle_force(
    constraint_self: float,
    gradient_self: np.ndarray,
    constraint_neighbor: float,
    support_radius: float,
    displacement_neighbor_minus_self: np.ndarray,
    rest_density: float,
) -> np.ndarray:
    """Reference two-particle IPBF force with neighbor constraint contributions."""
    if rest_density <= 0.0:
        return np.zeros(3, dtype=np.float32)

    neighbor_term = (
        constraint_neighbor
        * kernel_gradient_contribution(1.0, support_radius, displacement_neighbor_minus_self)
        / rest_density
    )
    return (-constraint_self * gradient_self + neighbor_term).astype(np.float32)


@wp.kernel
def evaluate_kernel_interface(
    displacement: wp.array(dtype=wp.vec3),
    support_radius: float,
    value: wp.array(dtype=float),
    gradient: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
):
    """Evaluate the current kernel interface for a set of query displacements."""
    tid = wp.tid()
    disp = displacement[tid]
    value[tid] = kernel_value(wp.dot(disp, disp), support_radius)
    gradient[tid] = kernel_gradient(disp, support_radius)
    hessian[tid] = kernel_hessian(disp, support_radius)


def run_two_particle_ipbf_step(
    device,
    *,
    iterations: int,
    relaxation: float,
    rest_density: float = 1.0,
    support_radius: float = 0.3,
    regularization: float = 1.0e-6,
):
    """Run one zero-gravity IPBF step for a symmetric two-particle setup."""
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

    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            hessian_regularization=regularization,
            iterations=iterations,
            relaxation=relaxation,
            use_constraint_clamp=False,
        ),
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)
    return state_1


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
        hessian_regularization=1.0e-6,
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
    expected_density = kernel_density_contribution(2.0, config.smoothing_radius, 0.0)
    expected_hessian = expected_ipbf_hessian(
        np.zeros(3, dtype=np.float32),
        mass=2.0,
        compliance=config.compliance,
        dt=dt,
        regularization=config.hessian_regularization,
    )

    np.testing.assert_allclose(state_1.ipbf.y.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.density.numpy(), np.array([expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(state_1.ipbf.constraint.numpy(), np.zeros(1, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.constraint_gradient.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.force.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.hessian.numpy()[0], expected_hessian, rtol=1e-5, atol=1e-5)
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
            iterations=0,
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
    np.testing.assert_allclose(state_0.ipbf.force.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


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
    regularization = 1.0e-6
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            hessian_regularization=regularization,
            iterations=0,
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
    force = state_1.ipbf.force.numpy()
    hessian = state_1.ipbf.hessian.numpy()

    distance = 0.1
    expected_density = kernel_density_contribution(1.0, support_radius, 0.0) + kernel_density_contribution(
        1.0, support_radius, distance
    )
    expected_constraint = expected_density / rest_density - 1.0
    expected_gradient_0 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_gradient_1 = -expected_gradient_0
    expected_constraint_hessian_0 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_constraint_hessian_1 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_neighbor_gradient_0 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_neighbor_gradient_1 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_neighbor_constraint_hessian_0 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_neighbor_constraint_hessian_1 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_0 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_0,
        expected_constraint,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_1 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_1,
        expected_constraint,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_hessian_0 = expected_ipbf_hessian(
        expected_gradient_0,
        mass=1.0,
        compliance=0.0,
        dt=0.05,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_0,
        neighbor_gradients=[expected_neighbor_gradient_0],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_0],
    )
    expected_hessian_1 = expected_ipbf_hessian(
        expected_gradient_1,
        mass=1.0,
        compliance=0.0,
        dt=0.05,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_1,
        neighbor_gradients=[expected_neighbor_gradient_1],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_1],
    )

    np.testing.assert_allclose(density, np.array([expected_density, expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        constraint,
        np.array([expected_constraint, expected_constraint], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(constraint_gradient[0], expected_gradient_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(constraint_gradient[1], expected_gradient_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force[0], expected_force_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force[1], expected_force_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hessian[0], expected_hessian_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hessian[1], expected_hessian_1, rtol=1e-5, atol=1e-5)
    test.assertGreater(float(constraint[0]), 0.0)


def test_ipbf_single_iteration_applies_relaxed_jacobi_update(test, device):
    rest_density = 1.0
    support_radius = 0.2
    relaxation = 0.5
    regularization = 1.0e-6
    dt = 0.05
    state_1 = run_two_particle_ipbf_step(
        device,
        iterations=1,
        relaxation=relaxation,
        rest_density=rest_density,
        support_radius=support_radius,
        regularization=regularization,
    )

    distance = 0.1
    expected_density = kernel_density_contribution(1.0, support_radius, 0.0) + kernel_density_contribution(
        1.0, support_radius, distance
    )
    expected_constraint = expected_density / rest_density - 1.0
    expected_gradient_0 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_gradient_1 = -expected_gradient_0
    expected_constraint_hessian_0 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_constraint_hessian_1 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_neighbor_gradient_0 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_neighbor_gradient_1 = kernel_gradient_contribution(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
    ) / rest_density
    expected_neighbor_constraint_hessian_0 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_neighbor_constraint_hessian_1 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_0 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_0,
        expected_constraint,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_1 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_1,
        expected_constraint,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_hessian_0 = expected_ipbf_hessian(
        expected_gradient_0,
        mass=1.0,
        compliance=0.0,
        dt=dt,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_0,
        neighbor_gradients=[expected_neighbor_gradient_0],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_0],
    )
    expected_hessian_1 = expected_ipbf_hessian(
        expected_gradient_1,
        mass=1.0,
        compliance=0.0,
        dt=dt,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_1,
        neighbor_gradients=[expected_neighbor_gradient_1],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_1],
    )
    expected_delta_0 = expected_ipbf_delta(expected_force_0, expected_hessian_0)
    expected_delta_1 = expected_ipbf_delta(expected_force_1, expected_hessian_1)
    expected_q_0 = np.array([-0.05, 1.0, 0.0], dtype=np.float32) + relaxation * expected_delta_0
    expected_q_1 = np.array([0.05, 1.0, 0.0], dtype=np.float32) + relaxation * expected_delta_1
    expected_qd_0 = relaxation * expected_delta_0 / dt
    expected_qd_1 = relaxation * expected_delta_1 / dt

    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy()[0], expected_delta_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy()[1], expected_delta_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_q_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[1], expected_q_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_q_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[1], expected_q_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_qd_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[1], expected_qd_1, rtol=1e-5, atol=1e-5)
    test.assertLess(float(state_1.particle_q.numpy()[0, 0]), -0.05)
    test.assertGreater(float(state_1.particle_q.numpy()[1, 0]), 0.05)


def test_ipbf_kernel_interface_matches_reference(test, device):
    support_radius = 0.3
    displacement = np.array([[0.05, -0.04, 0.02]], dtype=np.float32)

    displacement_wp = wp.array(displacement, dtype=wp.vec3, device=device)
    value_wp = wp.empty(1, dtype=float, device=device)
    gradient_wp = wp.empty(1, dtype=wp.vec3, device=device)
    hessian_wp = wp.empty(1, dtype=wp.mat33, device=device)

    wp.launch(
        evaluate_kernel_interface,
        dim=1,
        inputs=[displacement_wp, support_radius],
        outputs=[value_wp, gradient_wp, hessian_wp],
        device=device,
    )

    expected_value = kernel_density_contribution(1.0, support_radius, float(np.linalg.norm(displacement[0])))
    expected_gradient = kernel_gradient_contribution(1.0, support_radius, displacement[0])
    expected_hessian = kernel_hessian_reference(support_radius, displacement[0])

    np.testing.assert_allclose(value_wp.numpy()[0], expected_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gradient_wp.numpy()[0], expected_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessian_wp.numpy()[0], expected_hessian, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessian_wp.numpy()[0], hessian_wp.numpy()[0].T, rtol=1e-6, atol=1e-6)


def test_ipbf_multiple_iterations_reduce_constraint_magnitude(test, device):
    relaxation = 0.2
    state_0 = run_two_particle_ipbf_step(device, iterations=0, relaxation=relaxation)
    state_1 = run_two_particle_ipbf_step(device, iterations=1, relaxation=relaxation)
    state_2 = run_two_particle_ipbf_step(device, iterations=2, relaxation=relaxation)

    mean_abs_constraint_0 = float(np.mean(np.abs(state_0.ipbf.constraint.numpy())))
    mean_abs_constraint_1 = float(np.mean(np.abs(state_1.ipbf.constraint.numpy())))
    mean_abs_constraint_2 = float(np.mean(np.abs(state_2.ipbf.constraint.numpy())))

    test.assertGreater(mean_abs_constraint_0, mean_abs_constraint_1)
    test.assertGreater(mean_abs_constraint_1, mean_abs_constraint_2)
    test.assertGreater(float(np.abs(state_1.ipbf.delta_q.numpy()).max()), 0.0)
    test.assertGreater(float(np.abs(state_2.ipbf.delta_q.numpy()).max()), 0.0)
    test.assertLess(float(state_1.particle_q.numpy()[0, 0]), -0.05)
    test.assertLess(float(state_2.particle_q.numpy()[0, 0]), float(state_1.particle_q.numpy()[0, 0]))
    test.assertGreater(float(state_1.particle_q.numpy()[1, 0]), 0.05)
    test.assertGreater(float(state_2.particle_q.numpy()[1, 0]), float(state_1.particle_q.numpy()[1, 0]))


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
    np.testing.assert_allclose(state_1.ipbf.force.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
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
    "test_ipbf_single_iteration_applies_relaxed_jacobi_update",
    test_ipbf_single_iteration_applies_relaxed_jacobi_update,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_kernel_interface_matches_reference",
    test_ipbf_kernel_interface_matches_reference,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_multiple_iterations_reduce_constraint_magnitude",
    test_ipbf_multiple_iterations_reduce_constraint_magnitude,
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
