"""
IPBF solver kernels.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags


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
