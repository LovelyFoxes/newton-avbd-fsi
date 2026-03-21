"""IPBF solver kernels."""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def kernel_value(dist2: float, support_radius: float) -> float:
    """Evaluate the current scalar SPH kernel value.

    The current implementation uses the 3D poly6 kernel
    """
    if support_radius <= 0.0:
        return 0.0

    h2 = support_radius * support_radius
    if dist2 >= h2:
        return 0.0

    h3 = h2 * support_radius
    h9 = h3 * h3 * h3
    x = h2 - dist2
    return 315.0 / (64.0 * wp.pi * h9) * x * x * x


@wp.func
def kernel_gradient(displacement: wp.vec3, support_radius: float) -> wp.vec3:
    """Evaluate the spatial gradient of the current SPH kernel."""
    if support_radius <= 0.0:
        return wp.vec3(0.0)

    dist2 = wp.dot(displacement, displacement)
    h2 = support_radius * support_radius
    if dist2 >= h2 or dist2 == 0.0:
        return wp.vec3(0.0)

    h3 = h2 * support_radius
    h9 = h3 * h3 * h3
    x = h2 - dist2
    return displacement * (-945.0 / (32.0 * wp.pi * h9) * x * x)


@wp.func
def kernel_hessian(displacement: wp.vec3, support_radius: float) -> wp.mat33:
    """Evaluate the spatial Hessian of the current scalar SPH kernel."""
    if support_radius <= 0.0:
        return wp.mat33(0.0)

    dist2 = wp.dot(displacement, displacement)
    h2 = support_radius * support_radius
    if dist2 >= h2:
        return wp.mat33(0.0)

    h3 = h2 * support_radius
    h9 = h3 * h3 * h3
    x = h2 - dist2
    identity = wp.identity(n=3, dtype=float)

    return (
        -945.0 / (32.0 * wp.pi * h9) * x * x * identity
        + 945.0 / (8.0 * wp.pi * h9) * x * wp.outer(displacement, displacement)
    )


@wp.func
def diagonal_from_column_norms(matrix: wp.mat33) -> wp.mat33:
    """Return a diagonal matrix whose entries are the Euclidean column norms."""
    col0 = wp.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0] + matrix[2, 0] * matrix[2, 0])
    col1 = wp.sqrt(matrix[0, 1] * matrix[0, 1] + matrix[1, 1] * matrix[1, 1] + matrix[2, 1] * matrix[2, 1])
    col2 = wp.sqrt(matrix[0, 2] * matrix[0, 2] + matrix[1, 2] * matrix[1, 2] + matrix[2, 2] * matrix[2, 2])
    return wp.mat33(col0, 0.0, 0.0, 0.0, col1, 0.0, 0.0, 0.0, col2)


@wp.kernel
def predict_inertial_positions(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    y: wp.array(dtype=wp.vec3),
):
    """Compute inertial target positions for IPBF."""
    tid = wp.tid()

    x = particle_q[tid]
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        y[tid] = x
        return

    v = particle_qd[tid]
    inv_mass = particle_inv_mass[tid]
    f = particle_f[tid]

    world_idx = particle_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    if inv_mass > 0.0:
        a_star = f * inv_mass + world_g
    else:
        a_star = wp.vec3(0.0)

    y[tid] = x + dt * v + dt * dt * a_star


@wp.kernel
def initialize_guess_positions(
    y: wp.array(dtype=wp.vec3),
    x_guess: wp.array(dtype=wp.vec3),
    x_new: wp.array(dtype=wp.vec3),
):
    """Initialize the Jacobi guess buffers from inertial positions."""
    tid = wp.tid()
    yi = y[tid]
    x_guess[tid] = yi
    x_new[tid] = yi


@wp.kernel
def initialize_density_and_neighbor_count(
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    support_radius: float,
    density: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
):
    """Initialize density for states without a particle hash grid.

    This fallback is used for single-particle cases where Newton does not
    allocate a :class:`warp.HashGrid`.
    """
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        density[tid] = 0.0
        neighbor_count[tid] = 0
        return

    density[tid] = particle_mass[tid] * kernel_value(0.0, support_radius)
    neighbor_count[tid] = 0


@wp.kernel
def initialize_constraint_and_gradient(
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    rest_density: float,
    use_constraint_clamp: int,
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
):
    """Initialize constraint values for states without a particle hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        constraint[tid] = 0.0
        constraint_gradient[tid] = wp.vec3(0.0)
        return

    c = density[tid] / rest_density - 1.0
    if use_constraint_clamp != 0 and c < 0.0:
        c = 0.0

    constraint[tid] = c
    constraint_gradient[tid] = wp.vec3(0.0)


@wp.kernel
def compute_density_and_neighbor_count(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    support_radius: float,
    density: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
):
    """Compute SPH density estimates and neighbor counts from the particle hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        density[tid] = 0.0
        neighbor_count[tid] = 0
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    rho = float(0.0)
    count = int(0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        dist = xi - particle_q[index]
        dist2 = wp.dot(dist, dist)
        kernel = kernel_value(dist2, support_radius)
        if kernel <= 0.0:
            continue

        rho += particle_mass[index] * kernel
        if index != tid:
            count += 1

    density[tid] = rho
    neighbor_count[tid] = count


@wp.kernel
def compute_constraint_and_gradient(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    density: wp.array(dtype=float),
    rest_density: float,
    support_radius: float,
    use_constraint_clamp: int,
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
):
    """Compute density constraints and their local gradients from the hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        constraint[tid] = 0.0
        constraint_gradient[tid] = wp.vec3(0.0)
        return

    c = density[tid] / rest_density - 1.0
    if use_constraint_clamp != 0 and c < 0.0:
        constraint[tid] = 0.0
        constraint_gradient[tid] = wp.vec3(0.0)
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    grad = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        displacement = xi - particle_q[index]
        grad += particle_mass[index] * kernel_gradient(displacement, support_radius)

    constraint[tid] = c
    constraint_gradient[tid] = grad / rest_density


@wp.kernel
def compute_force(
    grid: wp.uint64,
    x_guess: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
    rest_density: float,
    support_radius: float,
    compliance: float,
    dt: float,
    force: wp.array(dtype=wp.vec3),
):
    """Compute the local IPBF force term used by the per-particle Newton step.

    The current implementation includes the self contribution from ``C_i`` and
    the neighboring constraint contributions that depend on ``x_i``.
    """
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or dt <= 0.0:
        force[tid] = wp.vec3(0.0)
        return

    inertia_scale = compliance * particle_mass[tid] / (dt * dt)
    inertial_force = -inertia_scale * (x_guess[tid] - y[tid])
    pressure_force = -constraint[tid] * constraint_gradient[tid]

    if rest_density > 0.0:
        xi = x_guess[tid]
        world_i = particle_world[tid]
        inv_rest_density = 1.0 / rest_density

        query = wp.hash_grid_query(grid, xi, support_radius)
        index = int(0)

        while wp.hash_grid_query_next(query, index):
            if index == tid or (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
                continue

            world_j = particle_world[index]
            if world_i >= 0 and world_j >= 0 and world_i != world_j:
                continue

            cj = constraint[index]
            if cj == 0.0:
                continue

            displacement = x_guess[index] - xi
            pressure_force += cj * particle_mass[tid] * inv_rest_density * kernel_gradient(displacement, support_radius)

    force[tid] = inertial_force + pressure_force


@wp.kernel
def compute_force_without_grid(
    x_guess: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
    compliance: float,
    dt: float,
    force: wp.array(dtype=wp.vec3),
):
    """Compute the local IPBF force term for cases without a hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or dt <= 0.0:
        force[tid] = wp.vec3(0.0)
        return

    inertia_scale = compliance * particle_mass[tid] / (dt * dt)
    inertial_force = -inertia_scale * (x_guess[tid] - y[tid])
    pressure_force = -constraint[tid] * constraint_gradient[tid]

    force[tid] = inertial_force + pressure_force


@wp.kernel
def compute_hessian(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
    rest_density: float,
    support_radius: float,
    compliance: float,
    dt: float,
    regularization: float,
    hessian: wp.array(dtype=wp.mat33),
):
    """Compute a stable IPBF Hessian approximation for the local solve.

    The current approximation keeps the Gauss-Newton term
    ``grad(C_i) grad(C_i)^T`` and adds a positive diagonal approximation of the
    second-order term based on the column norms of ``∇²C_i``.
    """
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        hessian[tid] = wp.mat33(0.0)
        return

    inertia_scale = float(0.0)
    if dt > 0.0:
        inertia_scale = compliance * particle_mass[tid] / (dt * dt)

    grad = constraint_gradient[tid]
    identity = wp.identity(n=3, dtype=float)
    h = (inertia_scale + regularization) * identity + wp.outer(grad, grad)

    if rest_density <= 0.0 or constraint[tid] == 0.0:
        hessian[tid] = h
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    constraint_hessian = wp.mat33(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        displacement = xi - particle_q[index]
        constraint_hessian += particle_mass[index] * kernel_hessian(displacement, support_radius)

    constraint_hessian = constraint_hessian / rest_density
    h += wp.abs(constraint[tid]) * diagonal_from_column_norms(constraint_hessian)
    hessian[tid] = h


@wp.kernel
def compute_hessian_without_grid(
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
    rest_density: float,
    support_radius: float,
    compliance: float,
    dt: float,
    regularization: float,
    hessian: wp.array(dtype=wp.mat33),
):
    """Compute the local Hessian approximation for cases without a hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        hessian[tid] = wp.mat33(0.0)
        return

    inertia_scale = float(0.0)
    if dt > 0.0:
        inertia_scale = compliance * particle_mass[tid] / (dt * dt)

    grad = constraint_gradient[tid]
    identity = wp.identity(n=3, dtype=float)
    h = (inertia_scale + regularization) * identity + wp.outer(grad, grad)

    if rest_density > 0.0 and constraint[tid] != 0.0:
        constraint_hessian = particle_mass[tid] * kernel_hessian(wp.vec3(0.0), support_radius) / rest_density
        h += wp.abs(constraint[tid]) * diagonal_from_column_norms(constraint_hessian)

    hessian[tid] = h


@wp.kernel
def solve_local_system(
    particle_flags: wp.array(dtype=wp.int32),
    force: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
    delta_q: wp.array(dtype=wp.vec3),
):
    """Solve the per-particle 3x3 linear system for the local position update."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        delta_q[tid] = wp.vec3(0.0)
        return

    h = hessian[tid]
    if abs(wp.determinant(h)) <= 1.0e-8:
        delta_q[tid] = wp.vec3(0.0)
        return

    delta_q[tid] = wp.inverse(h) * force[tid]


@wp.kernel
def apply_relaxed_jacobi_update(
    particle_flags: wp.array(dtype=wp.int32),
    x_guess: wp.array(dtype=wp.vec3),
    delta_q: wp.array(dtype=wp.vec3),
    relaxation: float,
    x_new: wp.array(dtype=wp.vec3),
):
    """Apply a relaxed Jacobi position update to form the next iterate."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        x_new[tid] = x_guess[tid]
        return

    x_new[tid] = x_guess[tid] + relaxation * delta_q[tid]


@wp.kernel
def update_velocity_from_positions(
    x_new: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Reconstruct particle velocities from the new and previous positions."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        particle_qd[tid] = wp.vec3(0.0)
        return

    particle_qd[tid] = (x_new[tid] - x_old[tid]) / dt
