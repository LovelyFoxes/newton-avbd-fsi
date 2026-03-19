"""IPBF solver kernels."""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def poly6_kernel(dist2: float, support_radius: float) -> float:
    """Evaluate the 3D poly6 SPH kernel with compact support ``support_radius``."""
    if support_radius <= 0.0:
        return 0.0

    h2 = support_radius * support_radius
    if dist2 >= h2:
        return 0.0

    h3 = h2 * support_radius
    h9 = h3 * h3 * h3
    x = h2 - dist2
    return 315.0 / (64.0 * wp.pi * h9) * x * x * x


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

    density[tid] = particle_mass[tid] * poly6_kernel(0.0, support_radius)
    neighbor_count[tid] = 0


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
        kernel = poly6_kernel(dist2, support_radius)
        if kernel <= 0.0:
            continue

        rho += particle_mass[index] * kernel
        if index != tid:
            count += 1

    density[tid] = rho
    neighbor_count[tid] = count


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
