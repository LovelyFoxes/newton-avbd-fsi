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

"""IPBF solver kernels."""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point

KERNEL_FAMILY_CUBIC_SPLINE = 0
KERNEL_FAMILY_POLY6 = 1
_BOUNDARY_SAMPLE_ACTIVE = wp.constant(1)
_BOUNDARY_SAMPLE_STATIC = wp.constant(1 << 1)


@wp.func
def kernel_value_cubic_spline(dist2: float, support_radius: float) -> float:
    """Evaluate the 3D cubic spline kernel value."""
    if support_radius <= 0.0:
        return 0.0

    r = wp.sqrt(dist2)
    if r >= support_radius:
        return 0.0

    inv_support_radius = 1.0 / support_radius
    q = 2.0 * r * inv_support_radius
    normalization = 8.0 * inv_support_radius * inv_support_radius * inv_support_radius / wp.pi

    if q < 1.0:
        return normalization * (1.0 - 1.5 * q * q + 0.75 * q * q * q)

    two_minus_q = 2.0 - q
    return normalization * 0.25 * two_minus_q * two_minus_q * two_minus_q


@wp.func
def kernel_value_poly6(dist2: float, support_radius: float) -> float:
    """Evaluate the 3D poly6 kernel value."""
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
def kernel_gradient_cubic_spline(displacement: wp.vec3, support_radius: float) -> wp.vec3:
    """Evaluate the spatial gradient of the 3D cubic spline kernel."""
    if support_radius <= 0.0:
        return wp.vec3(0.0)

    dist2 = wp.dot(displacement, displacement)
    if dist2 == 0.0:
        return wp.vec3(0.0)

    r = wp.sqrt(dist2)
    if r >= support_radius:
        return wp.vec3(0.0)

    inv_support_radius = 1.0 / support_radius
    q = 2.0 * r * inv_support_radius
    normalization = 8.0 * inv_support_radius * inv_support_radius * inv_support_radius / wp.pi
    q_scale = 2.0 * inv_support_radius

    dW_dr = float(0.0)
    if q < 1.0:
        dW_dr = normalization * q_scale * (-3.0 * q + 2.25 * q * q)
    else:
        two_minus_q = 2.0 - q
        dW_dr = normalization * q_scale * (-0.75 * two_minus_q * two_minus_q)

    return displacement * (dW_dr / r)


@wp.func
def kernel_gradient_poly6(displacement: wp.vec3, support_radius: float) -> wp.vec3:
    """Evaluate the spatial gradient of the 3D poly6 kernel."""
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
def kernel_hessian_cubic_spline(displacement: wp.vec3, support_radius: float) -> wp.mat33:
    """Evaluate the spatial Hessian of the 3D cubic spline kernel."""
    if support_radius <= 0.0:
        return wp.mat33(0.0)

    dist2 = wp.dot(displacement, displacement)
    r = wp.sqrt(dist2)
    if r >= support_radius:
        return wp.mat33(0.0)

    inv_support_radius = 1.0 / support_radius
    q = 2.0 * r * inv_support_radius
    normalization = 8.0 * inv_support_radius * inv_support_radius * inv_support_radius / wp.pi
    q_scale = 2.0 * inv_support_radius
    q_scale2 = q_scale * q_scale
    identity = wp.identity(n=3, dtype=float)

    if dist2 == 0.0:
        return normalization * q_scale2 * (-3.0) * identity

    dW_dr = float(0.0)
    d2W_dr2 = float(0.0)
    if q < 1.0:
        dW_dr = normalization * q_scale * (-3.0 * q + 2.25 * q * q)
        d2W_dr2 = normalization * q_scale2 * (-3.0 + 4.5 * q)
    else:
        two_minus_q = 2.0 - q
        dW_dr = normalization * q_scale * (-0.75 * two_minus_q * two_minus_q)
        d2W_dr2 = normalization * q_scale2 * (1.5 * two_minus_q)

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    outer_disp = wp.outer(displacement, displacement)

    return dW_dr * inv_r * identity + (d2W_dr2 - dW_dr * inv_r) * inv_r2 * outer_disp


@wp.func
def kernel_hessian_poly6(displacement: wp.vec3, support_radius: float) -> wp.mat33:
    """Evaluate the spatial Hessian of the 3D poly6 kernel."""
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

    return -945.0 / (32.0 * wp.pi * h9) * x * x * identity + 945.0 / (8.0 * wp.pi * h9) * x * wp.outer(
        displacement, displacement
    )


@wp.func
def kernel_value(dist2: float, support_radius: float, kernel_family: int) -> float:
    """Evaluate the selected scalar SPH kernel value."""
    if kernel_family == KERNEL_FAMILY_POLY6:
        return kernel_value_poly6(dist2, support_radius)

    return kernel_value_cubic_spline(dist2, support_radius)


@wp.func
def kernel_gradient(displacement: wp.vec3, support_radius: float, kernel_family: int) -> wp.vec3:
    """Evaluate the spatial gradient of the selected SPH kernel."""
    if kernel_family == KERNEL_FAMILY_POLY6:
        return kernel_gradient_poly6(displacement, support_radius)

    return kernel_gradient_cubic_spline(displacement, support_radius)


@wp.func
def kernel_hessian(displacement: wp.vec3, support_radius: float, kernel_family: int) -> wp.mat33:
    """Evaluate the spatial Hessian of the selected scalar SPH kernel."""
    if kernel_family == KERNEL_FAMILY_POLY6:
        return kernel_hessian_poly6(displacement, support_radius)

    return kernel_hessian_cubic_spline(displacement, support_radius)


@wp.func
def diagonal_from_column_norms(matrix: wp.mat33) -> wp.mat33:
    """Return a diagonal matrix whose entries are the Euclidean column norms."""
    col0 = wp.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0] + matrix[2, 0] * matrix[2, 0])
    col1 = wp.sqrt(matrix[0, 1] * matrix[0, 1] + matrix[1, 1] * matrix[1, 1] + matrix[2, 1] * matrix[2, 1])
    col2 = wp.sqrt(matrix[0, 2] * matrix[0, 2] + matrix[1, 2] * matrix[1, 2] + matrix[2, 2] * matrix[2, 2])
    return wp.mat33(col0, 0.0, 0.0, 0.0, col1, 0.0, 0.0, 0.0, col2)


@wp.func
def equivalent_force_from_position_delta(
    position_delta: wp.vec3,
    particle_mass: float,
    dt: float,
    reaction_relaxation: float,
) -> wp.vec3:
    """Convert a position correction into an equivalent average reaction force."""
    if dt <= 0.0 or reaction_relaxation == 0.0:
        return wp.vec3(0.0)

    return -reaction_relaxation * particle_mass * position_delta / (dt * dt)


@wp.func
def equivalent_force_from_velocity_delta(
    velocity_delta: wp.vec3,
    particle_mass: float,
    dt: float,
    reaction_relaxation: float,
) -> wp.vec3:
    """Convert a velocity correction into an equivalent average reaction force."""
    if dt <= 0.0 or reaction_relaxation == 0.0:
        return wp.vec3(0.0)

    return -reaction_relaxation * particle_mass * velocity_delta / dt


@wp.func
def body_reaction_torque_from_world_point(
    world_point: wp.vec3,
    reaction_force: wp.vec3,
    body_transform: wp.transform,
    body_com_local: wp.vec3,
) -> wp.vec3:
    """Compute torque from a force applied at a world-space point on a body."""
    com_world = wp.transform_point(body_transform, body_com_local)
    return wp.cross(world_point - com_world, reaction_force)


@wp.func
def boundary_density_pressure_weight(boundary_flag: int, static_boundary_weight: float) -> float:
    """Return the diagnostic density/pressure weight for a boundary sample."""
    if (boundary_flag & _BOUNDARY_SAMPLE_STATIC) != 0:
        return static_boundary_weight

    return 1.0


@wp.func
def boundary_triangle_shell_weight(
    boundary_triangle: int,
    boundary_normal: wp.vec3,
    displacement: wp.vec3,
    boundary_wet_weight: float,
) -> float:
    """Return wet-side one-sided support weight for deformable shell samples."""
    if boundary_triangle < 0:
        return 1.0

    if boundary_wet_weight <= 0.0:
        return 0.0

    if wp.dot(displacement, boundary_normal) >= 0.0:
        return boundary_wet_weight

    return 0.0


@wp.kernel
def update_boundary_triangle_wet_weights(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    support_radius: float,
    boundary_wet_weight: wp.array(dtype=float),
):
    """Update a binary wet-side indicator for deformable triangle samples."""
    tid = wp.tid()

    if boundary_triangle[tid] < 0:
        boundary_wet_weight[tid] = 1.0
        return

    xi = boundary_x[tid]
    normal = boundary_normal[tid]

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        displacement = particle_q[index] - xi
        if wp.dot(displacement, normal) <= 0.0:
            continue

        boundary_wet_weight[tid] = 1.0
        return

    boundary_wet_weight[tid] = 0.0


@wp.kernel
def mask_ipbf_particle_flags(
    particle_flags: wp.array(dtype=wp.int32),
    fluid_particle_start: int,
    fluid_particle_count: int,
    ipbf_particle_flags: wp.array(dtype=wp.int32),
):
    """Build solver-local particle flags for the active IPBF fluid range."""
    tid = wp.tid()

    if tid < fluid_particle_start or tid >= fluid_particle_start + fluid_particle_count:
        ipbf_particle_flags[tid] = 0
        return

    ipbf_particle_flags[tid] = particle_flags[tid]


@wp.kernel
def restore_non_fluid_particle_state(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_qd_in: wp.array(dtype=wp.vec3),
    fluid_particle_start: int,
    fluid_particle_count: int,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Restore particle state outside the active IPBF fluid range."""
    tid = wp.tid()

    if tid >= fluid_particle_start and tid < fluid_particle_start + fluid_particle_count:
        return

    particle_q_out[tid] = particle_q_in[tid]
    particle_qd_out[tid] = particle_qd_in[tid]


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
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    support_radius: float,
    kernel_family: int,
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

    density[tid] = particle_mass[tid] * kernel_value(0.0, support_radius, kernel_family)
    neighbor_count[tid] = 0


@wp.kernel
def initialize_density_and_neighbor_count_with_boundary(
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    boundary_wet_weight: wp.array(dtype=float),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    density: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
    density_boundary: wp.array(dtype=float),
    boundary_neighbor_count: wp.array(dtype=wp.int32),
):
    """Initialize density including boundary samples when no particle grid exists."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        density[tid] = 0.0
        neighbor_count[tid] = 0
        density_boundary[tid] = 0.0
        boundary_neighbor_count[tid] = 0
        return

    xi = particle_q[tid]
    rho = particle_mass[tid] * kernel_value(0.0, support_radius, kernel_family)
    rho_boundary = float(0.0)
    boundary_count = int(0)

    query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        boundary_flag = boundary_flags[index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        dist = xi - boundary_x[index]
        weight = weight * boundary_triangle_shell_weight(
            boundary_triangle[index], boundary_normal[index], dist, boundary_wet_weight[index]
        )
        if weight == 0.0:
            continue
        dist2 = wp.dot(dist, dist)
        kernel = kernel_value(dist2, support_radius, kernel_family)
        if kernel <= 0.0:
            continue

        rho_boundary += weight * rest_density * boundary_volume[index] * kernel
        boundary_count += 1

    density[tid] = rho + rho_boundary
    neighbor_count[tid] = boundary_count
    density_boundary[tid] = rho_boundary
    boundary_neighbor_count[tid] = boundary_count


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
def initialize_constraint_and_gradient_with_boundary(
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    boundary_wet_weight: wp.array(dtype=float),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    density: wp.array(dtype=float),
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    use_constraint_clamp: int,
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
):
    """Initialize constraints including boundary gradients when no particle grid exists."""
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
    grad = wp.vec3(0.0)

    query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        boundary_flag = boundary_flags[index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        displacement = xi - boundary_x[index]
        weight = weight * boundary_triangle_shell_weight(
            boundary_triangle[index],
            boundary_normal[index],
            displacement,
            boundary_wet_weight[index],
        )
        if weight == 0.0:
            continue
        grad += (
            weight
            * rest_density
            * boundary_volume[index]
            * kernel_gradient(displacement, support_radius, kernel_family)
        )

    constraint[tid] = c
    constraint_gradient[tid] = grad / rest_density


@wp.kernel
def compute_density_and_neighbor_count(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    support_radius: float,
    kernel_family: int,
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
        kernel = kernel_value(dist2, support_radius, kernel_family)
        if kernel <= 0.0:
            continue

        rho += particle_mass[index] * kernel
        if index != tid:
            count += 1

    density[tid] = rho
    neighbor_count[tid] = count


@wp.kernel
def compute_density_and_neighbor_count_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    boundary_wet_weight: wp.array(dtype=float),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    density: wp.array(dtype=float),
    neighbor_count: wp.array(dtype=wp.int32),
    density_boundary: wp.array(dtype=float),
    boundary_neighbor_count: wp.array(dtype=wp.int32),
):
    """Compute SPH density estimates with Akinci-style boundary samples."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        density[tid] = 0.0
        neighbor_count[tid] = 0
        density_boundary[tid] = 0.0
        boundary_neighbor_count[tid] = 0
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    rho = float(0.0)
    fluid_count = int(0)

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
        kernel = kernel_value(dist2, support_radius, kernel_family)
        if kernel <= 0.0:
            continue

        rho += particle_mass[index] * kernel
        if index != tid:
            fluid_count += 1

    rho_boundary = float(0.0)
    boundary_count = int(0)
    boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    boundary_index = int(0)

    while wp.hash_grid_query_next(boundary_query, boundary_index):
        boundary_flag = boundary_flags[boundary_index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        dist = xi - boundary_x[boundary_index]
        weight = weight * boundary_triangle_shell_weight(
            boundary_triangle[boundary_index],
            boundary_normal[boundary_index],
            dist,
            boundary_wet_weight[boundary_index],
        )
        if weight == 0.0:
            continue
        dist2 = wp.dot(dist, dist)
        kernel = kernel_value(dist2, support_radius, kernel_family)
        if kernel <= 0.0:
            continue

        rho_boundary += weight * rest_density * boundary_volume[boundary_index] * kernel
        boundary_count += 1

    density[tid] = rho + rho_boundary
    neighbor_count[tid] = fluid_count + boundary_count
    density_boundary[tid] = rho_boundary
    boundary_neighbor_count[tid] = boundary_count


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
    kernel_family: int,
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
        grad += particle_mass[index] * kernel_gradient(displacement, support_radius, kernel_family)

    constraint[tid] = c
    constraint_gradient[tid] = grad / rest_density


@wp.kernel
def compute_constraint_and_gradient_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    boundary_wet_weight: wp.array(dtype=float),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    density: wp.array(dtype=float),
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    use_constraint_clamp: int,
    constraint: wp.array(dtype=float),
    constraint_gradient: wp.array(dtype=wp.vec3),
):
    """Compute density constraints and gradients including boundary samples."""
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
        grad += particle_mass[index] * kernel_gradient(displacement, support_radius, kernel_family)

    boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    boundary_index = int(0)

    while wp.hash_grid_query_next(boundary_query, boundary_index):
        boundary_flag = boundary_flags[boundary_index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        displacement = xi - boundary_x[boundary_index]
        weight = weight * boundary_triangle_shell_weight(
            boundary_triangle[boundary_index],
            boundary_normal[boundary_index],
            displacement,
            boundary_wet_weight[boundary_index],
        )
        if weight == 0.0:
            continue
        grad += (
            weight
            * rest_density
            * boundary_volume[boundary_index]
            * kernel_gradient(displacement, support_radius, kernel_family)
        )

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
    kernel_family: int,
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
            pressure_force += (
                cj
                * particle_mass[tid]
                * inv_rest_density
                * kernel_gradient(displacement, support_radius, kernel_family)
            )

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
    kernel_family: int,
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

    if rest_density <= 0.0:
        hessian[tid] = h
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    constraint_hessian = wp.mat33(0.0)
    inv_rest_density = 1.0 / rest_density

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        displacement = xi - particle_q[index]
        constraint_hessian += particle_mass[index] * kernel_hessian(displacement, support_radius, kernel_family)
        if index != tid:
            neighbor_gradient = (
                particle_mass[tid] * inv_rest_density * kernel_gradient(-displacement, support_radius, kernel_family)
            )
            h += wp.outer(neighbor_gradient, neighbor_gradient)
            if constraint[index] != 0.0:
                neighbor_constraint_hessian = (
                    particle_mass[tid] * inv_rest_density * kernel_hessian(-displacement, support_radius, kernel_family)
                )
                h += wp.abs(constraint[index]) * diagonal_from_column_norms(neighbor_constraint_hessian)

    constraint_hessian = constraint_hessian * inv_rest_density
    if constraint[tid] != 0.0:
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
    kernel_family: int,
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
        constraint_hessian = particle_mass[tid] * kernel_hessian(wp.vec3(0.0), support_radius, kernel_family)

        constraint_hessian = constraint_hessian / rest_density
        h += wp.abs(constraint[tid]) * diagonal_from_column_norms(constraint_hessian)

    hessian[tid] = h


@wp.kernel
def accumulate_boundary_pressure_reaction(
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    constraint: wp.array(dtype=float),
    hessian: wp.array(dtype=wp.mat33),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_body: wp.array(dtype=wp.int32),
    boundary_triangle: wp.array(dtype=wp.int32),
    boundary_normal: wp.array(dtype=wp.vec3),
    boundary_wet_weight: wp.array(dtype=float),
    boundary_vertex0: wp.array(dtype=wp.int32),
    boundary_vertex1: wp.array(dtype=wp.int32),
    boundary_vertex2: wp.array(dtype=wp.int32),
    boundary_barycentric: wp.array(dtype=wp.vec3),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    support_radius: float,
    kernel_family: int,
    static_boundary_weight: float,
    dt: float,
    solve_relaxation: float,
    reaction_relaxation: float,
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    sample_force: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
    vertex_pressure_force: wp.array(dtype=wp.vec3),
    body_force: wp.array(dtype=wp.vec3),
    body_torque: wp.array(dtype=wp.vec3),
):
    """Accumulate equal-opposite body reaction from boundary pressure gradients."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    if dt <= 0.0 or solve_relaxation == 0.0 or reaction_relaxation == 0.0:
        return

    c = constraint[tid]
    if c <= 0.0:
        return

    xi = particle_q[tid]
    inv_h = wp.inverse(hessian[tid])
    query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    boundary_index = int(0)

    while wp.hash_grid_query_next(query, boundary_index):
        boundary_flag = boundary_flags[boundary_index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        displacement = xi - boundary_x[boundary_index]
        weight = weight * boundary_triangle_shell_weight(
            boundary_triangle[boundary_index],
            boundary_normal[boundary_index],
            displacement,
            boundary_wet_weight[boundary_index],
        )
        if weight == 0.0:
            continue
        boundary_grad = (
            weight * boundary_volume[boundary_index] * kernel_gradient(displacement, support_radius, kernel_family)
        )
        if wp.dot(boundary_grad, boundary_grad) == 0.0:
            continue

        pressure_rhs = -c * boundary_grad
        pressure_delta = solve_relaxation * (inv_h * pressure_rhs)
        if wp.dot(pressure_delta, pressure_delta) == 0.0:
            continue

        force_on_boundary = equivalent_force_from_position_delta(
            pressure_delta,
            particle_mass[tid],
            dt,
            reaction_relaxation,
        )
        wp.atomic_add(sample_force, boundary_index, force_on_boundary)

        if boundary_triangle[boundary_index] >= 0:
            barycentric = boundary_barycentric[boundary_index]
            denom = (
                barycentric[0] * barycentric[0]
                + barycentric[1] * barycentric[1]
                + barycentric[2] * barycentric[2]
            )
            if denom > 0.0:
                sample_delta = -reaction_relaxation * pressure_delta
                vertex_delta0 = (barycentric[0] / denom) * sample_delta
                vertex_delta1 = (barycentric[1] / denom) * sample_delta
                vertex_delta2 = (barycentric[2] / denom) * sample_delta
                wp.atomic_add(vertex_contact_delta, boundary_vertex0[boundary_index], vertex_delta0)
                wp.atomic_add(vertex_contact_delta, boundary_vertex1[boundary_index], vertex_delta1)
                wp.atomic_add(vertex_contact_delta, boundary_vertex2[boundary_index], vertex_delta2)
            wp.atomic_add(vertex_force, boundary_vertex0[boundary_index], barycentric[0] * force_on_boundary)
            wp.atomic_add(vertex_force, boundary_vertex1[boundary_index], barycentric[1] * force_on_boundary)
            wp.atomic_add(vertex_force, boundary_vertex2[boundary_index], barycentric[2] * force_on_boundary)
            wp.atomic_add(
                vertex_pressure_force,
                boundary_vertex0[boundary_index],
                barycentric[0] * force_on_boundary,
            )
            wp.atomic_add(
                vertex_pressure_force,
                boundary_vertex1[boundary_index],
                barycentric[1] * force_on_boundary,
            )
            wp.atomic_add(
                vertex_pressure_force,
                boundary_vertex2[boundary_index],
                barycentric[2] * force_on_boundary,
            )

        body_index = boundary_body[boundary_index]
        if body_index < 0:
            continue

        X_wb = body_q[body_index]
        torque = body_reaction_torque_from_world_point(
            boundary_x[boundary_index],
            force_on_boundary,
            X_wb,
            body_com[body_index],
        )

        wp.atomic_add(body_force, body_index, force_on_boundary)
        wp.atomic_add(body_torque, body_index, torque)


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
def project_particle_shape_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    relaxation: float,
):
    """Project particles out of penetrating particle-shape soft contacts."""
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    shape_index = contact_shape[tid]
    if particle_index < 0 or shape_index < 0:
        return

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    body_index = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]

    x = particle_q[particle_index]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    n = contact_normal[tid]
    c = wp.dot(n, x - bx) - particle_radius[particle_index]
    if c >= 0.0:
        return

    wp.atomic_add(particle_q, particle_index, -relaxation * c * n)


@wp.kernel
def project_particle_shape_contacts_with_reaction(
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    boundary_shape_sample_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    relaxation: float,
    dt: float,
    reaction_relaxation: float,
    particle_projection_delta: wp.array(dtype=wp.vec3),
    particle_projection_delta_total: wp.array(dtype=wp.vec3),
    body_force: wp.array(dtype=wp.vec3),
    body_torque: wp.array(dtype=wp.vec3),
):
    """Project particle-shape contacts and accumulate equal-opposite body reaction."""
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    shape_index = contact_shape[tid]
    if particle_index < 0 or shape_index < 0:
        return

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    body_index = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]

    x = particle_q[particle_index]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    n = contact_normal[tid]
    c = wp.dot(n, x - bx) - particle_radius[particle_index]
    if c >= 0.0:
        return

    correction = -relaxation * c * n
    wp.atomic_add(particle_q, particle_index, correction)
    wp.atomic_add(particle_projection_delta, particle_index, correction)
    wp.atomic_add(particle_projection_delta_total, particle_index, correction)

    if body_index < 0 or dt <= 0.0 or reaction_relaxation == 0.0 or boundary_shape_sample_count[shape_index] <= 0:
        return

    body_reaction = equivalent_force_from_position_delta(
        correction,
        particle_mass[particle_index],
        dt,
        reaction_relaxation,
    )
    torque = body_reaction_torque_from_world_point(
        bx,
        body_reaction,
        X_wb,
        body_com[body_index],
    )

    wp.atomic_add(body_force, body_index, body_reaction)
    wp.atomic_add(body_torque, body_index, torque)


@wp.func
def _lerp_vec3(a: wp.vec3, b: wp.vec3, t: float):
    return a + t * (b - a)


@wp.func
def _moving_triangle_plane_signed_distance(
    t: float,
    particle_x_prev: wp.vec3,
    particle_x_curr: wp.vec3,
    tri_x0_prev: wp.vec3,
    tri_x1_prev: wp.vec3,
    tri_x2_prev: wp.vec3,
    tri_x0_curr: wp.vec3,
    tri_x1_curr: wp.vec3,
    tri_x2_curr: wp.vec3,
):
    x = _lerp_vec3(particle_x_prev, particle_x_curr, t)
    x0 = _lerp_vec3(tri_x0_prev, tri_x0_curr, t)
    x1 = _lerp_vec3(tri_x1_prev, tri_x1_curr, t)
    x2 = _lerp_vec3(tri_x2_prev, tri_x2_curr, t)

    normal = wp.cross(x1 - x0, x2 - x0)
    normal_length = wp.length(normal)
    if normal_length <= 1.0e-8:
        return 0.0

    return wp.dot(normal / normal_length, x - x0)


@wp.func
def accumulate_particle_triangle_swept_contact_correction(
    tid: wp.int32,
    tri: wp.int32,
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_margin: float,
    relaxation: float,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
) -> int:
    wf = particle_inv_mass[tid]
    if wf <= 0.0:
        return 0

    x = particle_q[tid]
    contact_distance = particle_radius[tid] + contact_margin
    if contact_distance <= 0.0:
        return 0

    v0 = tri_indices[tri, 0]
    v1 = tri_indices[tri, 1]
    v2 = tri_indices[tri, 2]
    if tid == v0 or tid == v1 or tid == v2:
        return 0

    x_prev = particle_q_prev[tid]
    x0_prev = particle_q_prev[v0]
    x1_prev = particle_q_prev[v1]
    x2_prev = particle_q_prev[v2]
    x0 = particle_q[v0]
    x1 = particle_q[v1]
    x2 = particle_q[v2]

    normal_curr = wp.cross(x1 - x0, x2 - x0)
    normal_length = wp.length(normal_curr)
    if normal_length <= 0.0:
        return 0
    normal_curr = normal_curr / normal_length

    signed_prev = _moving_triangle_plane_signed_distance(
        0.0,
        x_prev,
        x,
        x0_prev,
        x1_prev,
        x2_prev,
        x0,
        x1,
        x2,
    )
    signed_curr = _moving_triangle_plane_signed_distance(
        1.0,
        x_prev,
        x,
        x0_prev,
        x1_prev,
        x2_prev,
        x0,
        x1,
        x2,
    )
    if signed_prev * signed_curr >= 0.0 or wp.abs(signed_prev) <= 1.0e-8 or wp.abs(signed_curr) <= 1.0e-8:
        return 0

    t_lo = 0.0
    t_hi = 1.0
    d_lo = signed_prev

    for _ in range(8):
        t_mid = 0.5 * (t_lo + t_hi)
        d_mid = _moving_triangle_plane_signed_distance(
            t_mid,
            x_prev,
            x,
            x0_prev,
            x1_prev,
            x2_prev,
            x0,
            x1,
            x2,
        )
        if d_lo * d_mid <= 0.0:
            t_hi = t_mid
        else:
            t_lo = t_mid
            d_lo = d_mid

    t = 0.5 * (t_lo + t_hi)
    hit = _lerp_vec3(x_prev, x, t)
    x0_hit = _lerp_vec3(x0_prev, x0, t)
    x1_hit = _lerp_vec3(x1_prev, x1, t)
    x2_hit = _lerp_vec3(x2_prev, x2, t)

    closest, barycentric, _feature_type = triangle_closest_point(x0_hit, x1_hit, x2_hit, hit)
    if wp.length(hit - closest) > contact_distance:
        return 0

    side = 1.0
    if signed_prev < 0.0:
        side = -1.0
    direction = side * normal_curr

    current_anchor = barycentric[0] * x0 + barycentric[1] * x1 + barycentric[2] * x2
    signed_curr_material = wp.dot(normal_curr, x - current_anchor)
    c = side * signed_curr_material - contact_distance
    if c >= 0.0:
        return 0

    w0 = particle_inv_mass[v0]
    w1 = particle_inv_mass[v1]
    w2 = particle_inv_mass[v2]
    b0 = barycentric[0]
    b1 = barycentric[1]
    b2 = barycentric[2]
    denom = wf + b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2
    if denom <= 0.0:
        return 0

    contact_lambda = -relaxation * c / denom
    fluid_delta = wf * contact_lambda * direction
    vertex_delta0 = -w0 * b0 * contact_lambda * direction
    vertex_delta1 = -w1 * b1 * contact_lambda * direction
    vertex_delta2 = -w2 * b2 * contact_lambda * direction

    wp.atomic_add(particle_contact_delta, tid, fluid_delta)
    wp.atomic_add(vertex_contact_delta, v0, vertex_delta0)
    wp.atomic_add(vertex_contact_delta, v1, vertex_delta1)
    wp.atomic_add(vertex_contact_delta, v2, vertex_delta2)
    wp.atomic_add(vertex_contact_delta_total, v0, vertex_delta0)
    wp.atomic_add(vertex_contact_delta_total, v1, vertex_delta1)
    wp.atomic_add(vertex_contact_delta_total, v2, vertex_delta2)

    if dt > 0.0:
        inv_dt2 = 1.0 / (dt * dt)
        wp.atomic_add(vertex_force, v0, particle_mass[v0] * vertex_delta0 * inv_dt2)
        wp.atomic_add(vertex_force, v1, particle_mass[v1] * vertex_delta1 * inv_dt2)
        wp.atomic_add(vertex_force, v2, particle_mass[v2] * vertex_delta2 * inv_dt2)

    return 1


@wp.func
def accumulate_particle_triangle_contact_correction(
    tid: wp.int32,
    tri: wp.int32,
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_margin: float,
    relaxation: float,
    continuous_enabled: int,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
):
    if continuous_enabled != 0:
        swept_contacted = accumulate_particle_triangle_swept_contact_correction(
            tid,
            tri,
            particle_q_prev,
            particle_q,
            particle_mass,
            particle_inv_mass,
            particle_radius,
            tri_indices,
            contact_margin,
            relaxation,
            dt,
            particle_contact_delta,
            vertex_contact_delta,
            vertex_contact_delta_total,
            vertex_force,
        )
        if swept_contacted != 0:
            return

    wf = particle_inv_mass[tid]
    if wf <= 0.0:
        return

    x = particle_q[tid]
    contact_distance = particle_radius[tid] + contact_margin
    if contact_distance <= 0.0:
        return

    v0 = tri_indices[tri, 0]
    v1 = tri_indices[tri, 1]
    v2 = tri_indices[tri, 2]
    if tid == v0 or tid == v1 or tid == v2:
        return

    x0 = particle_q[v0]
    x1 = particle_q[v1]
    x2 = particle_q[v2]

    normal = wp.cross(x1 - x0, x2 - x0)
    normal_length = wp.length(normal)
    if normal_length <= 0.0:
        return
    normal = normal / normal_length

    closest, barycentric, _feature_type = triangle_closest_point(x0, x1, x2, x)
    separation = x - closest
    distance = wp.length(separation)

    direction = normal
    if distance > 1.0e-8:
        direction = separation / distance
    else:
        signed_plane_distance = wp.dot(normal, x - x0)
        if signed_plane_distance < 0.0:
            direction = -normal
        distance = wp.abs(signed_plane_distance)

    c = distance - contact_distance
    if c >= 0.0:
        return

    w0 = particle_inv_mass[v0]
    w1 = particle_inv_mass[v1]
    w2 = particle_inv_mass[v2]
    b0 = barycentric[0]
    b1 = barycentric[1]
    b2 = barycentric[2]
    denom = wf + b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2
    if denom <= 0.0:
        return

    contact_lambda = -relaxation * c / denom
    fluid_delta = wf * contact_lambda * direction
    vertex_delta0 = -w0 * b0 * contact_lambda * direction
    vertex_delta1 = -w1 * b1 * contact_lambda * direction
    vertex_delta2 = -w2 * b2 * contact_lambda * direction

    wp.atomic_add(particle_contact_delta, tid, fluid_delta)
    wp.atomic_add(vertex_contact_delta, v0, vertex_delta0)
    wp.atomic_add(vertex_contact_delta, v1, vertex_delta1)
    wp.atomic_add(vertex_contact_delta, v2, vertex_delta2)
    wp.atomic_add(vertex_contact_delta_total, v0, vertex_delta0)
    wp.atomic_add(vertex_contact_delta_total, v1, vertex_delta1)
    wp.atomic_add(vertex_contact_delta_total, v2, vertex_delta2)

    if dt > 0.0:
        inv_dt2 = 1.0 / (dt * dt)
        wp.atomic_add(vertex_force, v0, particle_mass[v0] * vertex_delta0 * inv_dt2)
        wp.atomic_add(vertex_force, v1, particle_mass[v1] * vertex_delta1 * inv_dt2)
        wp.atomic_add(vertex_force, v2, particle_mass[v2] * vertex_delta2 * inv_dt2)


@wp.kernel
def accumulate_particle_triangle_contact_corrections(
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_triangle_indices: wp.array(dtype=wp.int32),
    contact_triangle_count: int,
    contact_margin: float,
    relaxation: float,
    continuous_enabled: int,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
):
    """Accumulate symmetric particle-triangle contact corrections by scanning all sampled triangles."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    if relaxation == 0.0:
        return

    for contact_triangle_index in range(contact_triangle_count):
        tri = contact_triangle_indices[contact_triangle_index]
        accumulate_particle_triangle_contact_correction(
            tid,
            tri,
            particle_q_prev,
            particle_q,
            particle_mass,
            particle_inv_mass,
            particle_radius,
            tri_indices,
            contact_margin,
            relaxation,
            continuous_enabled,
            dt,
            particle_contact_delta,
            vertex_contact_delta,
            vertex_contact_delta_total,
            vertex_force,
        )


@wp.kernel
def initialize_triangle_contact_pair_cache_reuse(
    pair_cache_valid: wp.array(dtype=wp.int32),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_displacement_max: wp.array(dtype=float),
):
    """Initialize device-side pair-cache reuse state for the current contact solve."""
    if wp.tid() != 0:
        return

    pair_cache_reuse[0] = pair_cache_valid[0]
    pair_cache_displacement_max[0] = 0.0


@wp.kernel
def accumulate_triangle_contact_pair_cache_fluid_displacement_max(
    pair_particle: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_capacity: int,
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_displacement_max: wp.array(dtype=float),
):
    """Accumulate the maximum displacement of cached fluid-pair particles since the snapshot."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    pair_total = wp.min(pair_count[0], pair_capacity)
    if tid >= pair_total:
        return

    particle_index = pair_particle[tid]
    if particle_index < 0:
        return

    displacement = wp.length(particle_q[particle_index] - pair_cache_particle_q[particle_index])
    wp.atomic_max(pair_cache_displacement_max, 0, displacement)


@wp.kernel
def accumulate_triangle_contact_pair_cache_triangle_displacement_max(
    contact_triangle_indices: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_displacement_max: wp.array(dtype=float),
):
    """Accumulate the maximum displacement of sampled cloth-triangle vertices since the snapshot."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    tri = contact_triangle_indices[tid]
    v0 = tri_indices[tri, 0]
    v1 = tri_indices[tri, 1]
    v2 = tri_indices[tri, 2]

    displacement0 = wp.length(particle_q[v0] - pair_cache_particle_q[v0])
    displacement1 = wp.length(particle_q[v1] - pair_cache_particle_q[v1])
    displacement2 = wp.length(particle_q[v2] - pair_cache_particle_q[v2])

    wp.atomic_max(pair_cache_displacement_max, 0, displacement0)
    wp.atomic_max(pair_cache_displacement_max, 0, displacement1)
    wp.atomic_max(pair_cache_displacement_max, 0, displacement2)


@wp.kernel
def update_triangle_contact_pair_cache_snapshot_if_active(
    particle_flags: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
):
    """Refresh cached snapshots only for active fluid particles when recollecting."""
    if pair_cache_reuse[0] != 0:
        return

    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    pair_cache_particle_q[tid] = particle_q[tid]


@wp.kernel
def update_triangle_contact_pair_cache_snapshot_for_sampled_triangles(
    contact_triangle_indices: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
):
    """Refresh cached snapshots for sampled cloth-triangle vertices when recollecting."""
    if pair_cache_reuse[0] != 0:
        return

    tid = wp.tid()
    tri = contact_triangle_indices[tid]
    v0 = tri_indices[tri, 0]
    v1 = tri_indices[tri, 1]
    v2 = tri_indices[tri, 2]

    pair_cache_particle_q[v0] = particle_q[v0]
    pair_cache_particle_q[v1] = particle_q[v1]
    pair_cache_particle_q[v2] = particle_q[v2]


@wp.kernel
def clear_triangle_contact_pair_cache_snapshot_if_inactive(
    particle_flags: wp.array(dtype=wp.int32),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
):
    """Clear cached snapshot entries for particles outside the active fluid set."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        pair_cache_particle_q[tid] = wp.vec3(0.0)


@wp.kernel
def clear_triangle_contact_pair_cache_reuse_for_inactive(
    particle_flags: wp.array(dtype=wp.int32),
    pair_particle: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_capacity: int,
    pair_cache_reuse: wp.array(dtype=wp.int32),
):
    """Invalidate reuse if a cached pair references a particle that is no longer active."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    pair_total = wp.min(pair_count[0], pair_capacity)
    if tid >= pair_total:
        return

    particle_index = pair_particle[tid]
    if particle_index < 0:
        return

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        pair_cache_reuse[0] = 0


@wp.kernel
def mark_triangle_contact_pair_cached_particles(
    pair_particle: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_capacity: int,
    pair_cache_reuse: wp.array(dtype=wp.int32),
    cached_particle_mask: wp.array(dtype=wp.int32),
):
    """Mark active fluid particles that already own cached compact triangle pairs."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    pair_total = wp.min(pair_count[0], pair_capacity)
    if tid >= pair_total:
        return

    particle_index = pair_particle[tid]
    if particle_index >= 0:
        cached_particle_mask[particle_index] = 1


@wp.kernel
def clear_triangle_contact_pair_cache_reuse_for_uncached_particles_near_bvh(
    triangle_contact_bvh: wp.uint64,
    particle_q_snapshot: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    cached_particle_mask: wp.array(dtype=wp.int32),
    contact_margin: float,
    pair_cache_skin: float,
    pair_cache_reuse: wp.array(dtype=wp.int32),
):
    """Invalidate reuse if an uncached active fluid particle enters triangle-BVH query range."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or cached_particle_mask[tid] != 0:
        return

    query_padding = particle_radius[tid] + contact_margin + 2.0 * pair_cache_skin
    if query_padding <= 0.0:
        return

    x_prev = particle_q_snapshot[tid]
    x = particle_q[tid]
    lower = wp.min(x_prev, x) - wp.vec3(query_padding)
    upper = wp.max(x_prev, x) + wp.vec3(query_padding)

    query = wp.bvh_query_aabb(triangle_contact_bvh, lower, upper)
    triangle_leaf = wp.int32(-1)
    if wp.bvh_query_next(query, triangle_leaf):
        pair_cache_reuse[0] = 0


@wp.kernel
def clear_triangle_contact_pair_cache_reuse_for_triangle_count_change(
    pair_count: wp.array(dtype=wp.int32),
    pair_cache_valid: wp.array(dtype=wp.int32),
    pair_cache_reuse: wp.array(dtype=wp.int32),
):
    """Invalidate reuse if the cached compact pair set is empty or invalid."""
    if wp.tid() != 0:
        return

    if pair_cache_valid[0] == 0 or pair_count[0] <= 0:
        pair_cache_reuse[0] = 0


@wp.kernel
def accumulate_triangle_contact_pair_cache_displacement_max(
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_displacement_max: wp.array(dtype=float),
):
    """Deprecated global displacement path kept for compatibility."""
    if pair_cache_reuse[0] == 0:
        return

    tid = wp.tid()
    displacement = wp.length(particle_q[tid] - pair_cache_particle_q[tid])
    wp.atomic_max(pair_cache_displacement_max, 0, displacement)


@wp.kernel
def finalize_triangle_contact_pair_cache_reuse(
    pair_cache_valid: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_cache_overflow: wp.array(dtype=wp.int32),
    pair_cache_displacement_max: wp.array(dtype=float),
    pair_cache_skin: float,
    pair_cache_reuse: wp.array(dtype=wp.int32),
):
    """Finalize whether cached compact contact pairs may be reused this solve."""
    if wp.tid() != 0:
        return

    if pair_cache_reuse[0] == 0:
        return

    reuse = 0
    if (
        pair_cache_valid[0] != 0
        and pair_count[0] > 0
        and pair_cache_overflow[0] == 0
        and pair_cache_skin > 0.0
        and pair_cache_displacement_max[0] <= pair_cache_skin
    ):
        reuse = 1

    pair_cache_reuse[0] = reuse


@wp.kernel
def prepare_triangle_contact_pair_collection(
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_overflow: wp.array(dtype=wp.int32),
):
    """Reset compact pair collection buffers when cache reuse is not active."""
    if wp.tid() != 0 or pair_cache_reuse[0] != 0:
        return

    pair_count[0] = 0
    pair_overflow[0] = 0


@wp.kernel
def collect_particle_triangle_contact_pairs_from_bvh(
    triangle_contact_bvh: wp.uint64,
    contact_triangle_indices: wp.array(dtype=wp.int32),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    contact_margin: float,
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_cache_skin: float,
    pair_capacity: int,
    pair_particle: wp.array(dtype=wp.int32),
    pair_triangle: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_overflow: wp.array(dtype=wp.int32),
):
    """Collect compact particle-triangle candidate pairs from a swept triangle BVH."""
    tid = wp.tid()

    if pair_cache_reuse[0] != 0:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    query_padding = particle_radius[tid] + contact_margin + 2.0 * pair_cache_skin
    if query_padding <= 0.0:
        return

    x_prev = particle_q_prev[tid]
    x = particle_q[tid]
    lower = wp.min(x_prev, x) - wp.vec3(query_padding)
    upper = wp.max(x_prev, x) + wp.vec3(query_padding)

    query = wp.bvh_query_aabb(triangle_contact_bvh, lower, upper)
    triangle_leaf = wp.int32(-1)

    while wp.bvh_query_next(query, triangle_leaf):
        pair_index = wp.atomic_add(pair_count, 0, 1)
        if pair_index < pair_capacity:
            pair_particle[pair_index] = tid
            pair_triangle[pair_index] = contact_triangle_indices[triangle_leaf]
        else:
            pair_overflow[0] = 1


@wp.kernel
def update_triangle_contact_pair_cache_snapshot(
    pair_cache_reuse: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    pair_cache_particle_q: wp.array(dtype=wp.vec3),
):
    """Refresh the cached particle snapshot when a new compact pair set is collected."""
    if pair_cache_reuse[0] != 0:
        return

    tid = wp.tid()
    pair_cache_particle_q[tid] = particle_q[tid]


@wp.kernel
def finalize_triangle_contact_pair_cache_collection(
    pair_cache_reuse: wp.array(dtype=wp.int32),
    pair_overflow: wp.array(dtype=wp.int32),
    pair_cache_valid: wp.array(dtype=wp.int32),
):
    """Commit the current compact pair collection as reusable cache state."""
    if wp.tid() != 0 or pair_cache_reuse[0] != 0:
        return

    pair_cache_valid[0] = 0 if pair_overflow[0] != 0 else 1


@wp.kernel
def accumulate_particle_triangle_contact_corrections_from_pairs(
    pair_particle: wp.array(dtype=wp.int32),
    pair_triangle: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_overflow: wp.array(dtype=wp.int32),
    pair_capacity: int,
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_margin: float,
    relaxation: float,
    continuous_enabled: int,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
):
    """Accumulate particle-triangle contact corrections from compact candidate pairs."""
    tid = wp.tid()

    if relaxation == 0.0:
        return
    if pair_overflow[0] != 0:
        return

    pair_total = wp.min(pair_count[0], pair_capacity)
    if tid >= pair_total:
        return

    particle_index = pair_particle[tid]
    tri = pair_triangle[tid]
    if particle_index < 0 or tri < 0:
        return

    accumulate_particle_triangle_contact_correction(
        particle_index,
        tri,
        particle_q_prev,
        particle_q,
        particle_mass,
        particle_inv_mass,
        particle_radius,
        tri_indices,
        contact_margin,
        relaxation,
        continuous_enabled,
        dt,
        particle_contact_delta,
        vertex_contact_delta,
        vertex_contact_delta_total,
        vertex_force,
        )


@wp.kernel
def accumulate_particle_triangle_contact_corrections_if_overflow(
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_triangle_indices: wp.array(dtype=wp.int32),
    contact_triangle_count: int,
    pair_overflow: wp.array(dtype=wp.int32),
    contact_margin: float,
    relaxation: float,
    continuous_enabled: int,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
):
    """Accumulate full-scan triangle contacts only when compact pair collection overflowed."""
    tid = wp.tid()

    if pair_overflow[0] == 0:
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    if relaxation == 0.0:
        return

    for contact_triangle_index in range(contact_triangle_count):
        tri = contact_triangle_indices[contact_triangle_index]
        accumulate_particle_triangle_contact_correction(
            tid,
            tri,
            particle_q_prev,
            particle_q,
            particle_mass,
            particle_inv_mass,
            particle_radius,
            tri_indices,
            contact_margin,
            relaxation,
            continuous_enabled,
            dt,
            particle_contact_delta,
            vertex_contact_delta,
            vertex_contact_delta_total,
            vertex_force,
        )


@wp.kernel
def accumulate_particle_triangle_contact_corrections_from_grid(
    triangle_contact_grid: wp.uint64,
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    contact_triangle_indices: wp.array(dtype=wp.int32),
    triangle_proxy_motion_max: wp.array(dtype=float),
    contact_search_radius: float,
    contact_margin: float,
    relaxation: float,
    continuous_enabled: int,
    dt: float,
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta_total: wp.array(dtype=wp.vec3),
    vertex_force: wp.array(dtype=wp.vec3),
):
    """Accumulate particle-triangle contact corrections from nearby triangle proxies."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    if relaxation == 0.0 or contact_search_radius <= 0.0:
        return

    query_center = particle_q[tid]
    query_radius = contact_search_radius
    if continuous_enabled != 0:
        travel = particle_q[tid] - particle_q_prev[tid]
        query_center = 0.5 * (particle_q[tid] + particle_q_prev[tid])
        query_radius = query_radius + 0.5 * wp.length(travel) + 0.5 * triangle_proxy_motion_max[0]

    query = wp.hash_grid_query(triangle_contact_grid, query_center, query_radius)
    triangle_proxy = int(0)

    while wp.hash_grid_query_next(query, triangle_proxy):
        tri = contact_triangle_indices[triangle_proxy]
        accumulate_particle_triangle_contact_correction(
            tid,
            tri,
            particle_q_prev,
            particle_q,
            particle_mass,
            particle_inv_mass,
            particle_radius,
            tri_indices,
            contact_margin,
            relaxation,
            continuous_enabled,
            dt,
            particle_contact_delta,
            vertex_contact_delta,
            vertex_contact_delta_total,
            vertex_force,
        )


@wp.kernel
def apply_particle_triangle_contact_deltas(
    particle_flags: wp.array(dtype=wp.int32),
    particle_contact_delta: wp.array(dtype=wp.vec3),
    vertex_contact_delta: wp.array(dtype=wp.vec3),
    apply_vertex_deltas: int,
    particle_q: wp.array(dtype=wp.vec3),
):
    """Apply accumulated particle-triangle contact deltas to positions."""
    tid = wp.tid()

    delta = wp.vec3(0.0)
    if (particle_flags[tid] & ParticleFlags.ACTIVE) != 0:
        delta = delta + particle_contact_delta[tid]
    if apply_vertex_deltas != 0:
        delta = delta + vertex_contact_delta[tid]

    particle_q[tid] = particle_q[tid] + delta


@wp.kernel
def initialize_particle_shape_boundary_velocity_projection(
    particle_flags: wp.array(dtype=wp.int32),
    projected_particle_qd: wp.array(dtype=wp.vec3),
    projected_contact_count: wp.array(dtype=int),
):
    """Initialize boundary-velocity projection accumulators."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        projected_particle_qd[tid] = wp.vec3(0.0)
        projected_contact_count[tid] = 0
        return

    projected_particle_qd[tid] = wp.vec3(0.0)
    projected_contact_count[tid] = 0


@wp.kernel
def accumulate_particle_shape_boundary_velocity_projection(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    tangential_damping: float,
    projected_particle_qd: wp.array(dtype=wp.vec3),
    projected_contact_count: wp.array(dtype=int),
):
    """Accumulate per-contact projected particle velocities against shape boundaries."""
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    if particle_index < 0:
        return

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    n = contact_normal[tid]
    body_v = contact_body_vel[tid]
    rel_v = particle_qd[particle_index] - body_v
    rel_v_n = wp.dot(rel_v, n)
    rel_v_t = rel_v - rel_v_n * n
    rel_v_projected = wp.max(rel_v_n, 0.0) * n + tangential_damping * rel_v_t
    projected_v = body_v + rel_v_projected

    wp.atomic_add(projected_particle_qd, particle_index, projected_v)
    wp.atomic_add(projected_contact_count, particle_index, 1)


@wp.kernel
def accumulate_particle_shape_boundary_velocity_projection_with_reaction(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    boundary_shape_sample_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    tangential_damping: float,
    dt: float,
    reaction_relaxation: float,
    projected_particle_qd: wp.array(dtype=wp.vec3),
    projected_contact_count: wp.array(dtype=int),
    body_force: wp.array(dtype=wp.vec3),
    body_torque: wp.array(dtype=wp.vec3),
):
    """Project boundary velocities and accumulate equal-opposite contact reaction."""
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    shape_index = contact_shape[tid]
    if particle_index < 0 or shape_index < 0:
        return

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    n = contact_normal[tid]
    body_v = contact_body_vel[tid]
    old_v = particle_qd[particle_index]
    rel_v = old_v - body_v
    rel_v_n = wp.dot(rel_v, n)
    rel_v_t = rel_v - rel_v_n * n
    rel_v_projected = wp.max(rel_v_n, 0.0) * n + tangential_damping * rel_v_t
    projected_v = body_v + rel_v_projected

    wp.atomic_add(projected_particle_qd, particle_index, projected_v)
    wp.atomic_add(projected_contact_count, particle_index, 1)

    body_index = shape_body[shape_index]
    if body_index < 0 or dt <= 0.0 or reaction_relaxation == 0.0 or boundary_shape_sample_count[shape_index] <= 0:
        return

    delta_v = projected_v - old_v
    if wp.dot(delta_v, delta_v) == 0.0:
        return

    body_reaction = equivalent_force_from_velocity_delta(
        delta_v,
        particle_mass[particle_index],
        dt,
        reaction_relaxation,
    )
    X_wb = body_q[body_index]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    torque = body_reaction_torque_from_world_point(
        bx,
        body_reaction,
        X_wb,
        body_com[body_index],
    )

    wp.atomic_add(body_force, body_index, body_reaction)
    wp.atomic_add(body_torque, body_index, torque)


@wp.kernel
def finalize_particle_shape_boundary_velocity_projection(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    projected_particle_qd: wp.array(dtype=wp.vec3),
    projected_contact_count: wp.array(dtype=int),
):
    """Average accumulated boundary-projected velocities back into the particle state."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        particle_qd[tid] = wp.vec3(0.0)
        return

    count = projected_contact_count[tid]
    if count > 0:
        particle_qd[tid] = projected_particle_qd[tid] / float(count)


@wp.kernel
def apply_xsph_velocity_smoothing(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    support_radius: float,
    kernel_family: int,
    xsph_coefficient: float,
    smoothed_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply an XSPH-style velocity smoothing step using particle neighbors."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or xsph_coefficient <= 0.0:
        smoothed_particle_qd[tid] = particle_qd[tid]
        return

    xi = particle_q[tid]
    vi = particle_qd[tid]
    world_i = particle_world[tid]
    correction = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index == tid or (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        rho_j = density[index]
        if rho_j <= 0.0:
            continue

        displacement = xi - particle_q[index]
        weight = kernel_value(wp.dot(displacement, displacement), support_radius, kernel_family)
        if weight <= 0.0:
            continue

        correction += (particle_mass[index] / rho_j) * (particle_qd[index] - vi) * weight

    smoothed_particle_qd[tid] = vi + xsph_coefficient * correction


@wp.kernel
def apply_xsph_velocity_smoothing_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_v: wp.array(dtype=wp.vec3),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    xsph_coefficient: float,
    xsph_boundary_coefficient: float,
    smoothed_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply XSPH smoothing with both fluid and boundary sample neighbors."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        smoothed_particle_qd[tid] = particle_qd[tid]
        return

    if xsph_coefficient <= 0.0 and xsph_boundary_coefficient <= 0.0:
        smoothed_particle_qd[tid] = particle_qd[tid]
        return

    xi = particle_q[tid]
    vi = particle_qd[tid]
    rho_i = density[tid]
    world_i = particle_world[tid]
    fluid_correction = wp.vec3(0.0)
    boundary_correction = wp.vec3(0.0)

    if xsph_coefficient > 0.0:
        query = wp.hash_grid_query(grid, xi, support_radius)
        index = int(0)

        while wp.hash_grid_query_next(query, index):
            if index == tid or (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
                continue

            world_j = particle_world[index]
            if world_i >= 0 and world_j >= 0 and world_i != world_j:
                continue

            rho_j = density[index]
            if rho_j <= 0.0:
                continue

            displacement = xi - particle_q[index]
            weight = kernel_value(wp.dot(displacement, displacement), support_radius, kernel_family)
            if weight <= 0.0:
                continue

            fluid_correction += (particle_mass[index] / rho_j) * (particle_qd[index] - vi) * weight

    if xsph_boundary_coefficient > 0.0 and rho_i > 0.0 and rest_density > 0.0:
        boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
        boundary_index = int(0)

        while wp.hash_grid_query_next(boundary_query, boundary_index):
            boundary_flag = boundary_flags[boundary_index]
            if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
                continue
            weight_scale = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
            if weight_scale == 0.0:
                continue

            displacement = xi - boundary_x[boundary_index]
            weight = kernel_value(wp.dot(displacement, displacement), support_radius, kernel_family)
            if weight <= 0.0:
                continue

            boundary_correction += (
                weight_scale
                * (rest_density * boundary_volume[boundary_index] / rho_i)
                * (boundary_v[boundary_index] - vi)
                * weight
            )

    smoothed_particle_qd[tid] = (
        vi + xsph_coefficient * fluid_correction + xsph_boundary_coefficient * boundary_correction
    )


@wp.kernel
def apply_xsph_velocity_smoothing_without_grid(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    xsph_coefficient: float,
    smoothed_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply XSPH-style smoothing for cases without a particle hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or xsph_coefficient <= 0.0:
        smoothed_particle_qd[tid] = particle_qd[tid]
        return

    vi = particle_qd[tid]
    smoothed_particle_qd[tid] = vi


@wp.kernel
def apply_viscosity_velocity_diffusion(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    support_radius: float,
    kernel_family: int,
    viscosity_coefficient: float,
    dt: float,
    diffused_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply a Morris-style SPH viscosity diffusion step to particle velocities."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or viscosity_coefficient <= 0.0 or dt <= 0.0:
        diffused_particle_qd[tid] = particle_qd[tid]
        return

    xi = particle_q[tid]
    vi = particle_qd[tid]
    world_i = particle_world[tid]
    h2 = support_radius * support_radius
    eps2 = 1.0e-2 * h2
    diffusion = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if index == tid or (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue

        rho_j = density[index]
        if rho_j <= 0.0:
            continue

        displacement = xi - particle_q[index]
        dist2 = wp.dot(displacement, displacement)
        if dist2 <= 0.0 or dist2 >= h2:
            continue

        grad = kernel_gradient(displacement, support_radius, kernel_family)
        laplace_weight = -wp.dot(displacement, grad) / (dist2 + eps2)
        if laplace_weight <= 0.0:
            continue

        diffusion += (particle_mass[index] / rho_j) * (particle_qd[index] - vi) * laplace_weight

    diffused_particle_qd[tid] = vi + 10.0 * viscosity_coefficient * dt * diffusion


@wp.kernel
def apply_viscosity_velocity_diffusion_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_v: wp.array(dtype=wp.vec3),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    viscosity_coefficient: float,
    viscosity_boundary_coefficient: float,
    dt: float,
    diffused_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply viscosity diffusion using both fluid and boundary sample neighbors."""
    tid = wp.tid()

    if (
        (particle_flags[tid] & ParticleFlags.ACTIVE) == 0
        or dt <= 0.0
        or (viscosity_coefficient <= 0.0 and viscosity_boundary_coefficient <= 0.0)
    ):
        diffused_particle_qd[tid] = particle_qd[tid]
        return

    xi = particle_q[tid]
    vi = particle_qd[tid]
    rho_i = density[tid]
    world_i = particle_world[tid]
    h2 = support_radius * support_radius
    eps2 = 1.0e-2 * h2
    fluid_diffusion = wp.vec3(0.0)
    boundary_diffusion = wp.vec3(0.0)

    if viscosity_coefficient > 0.0:
        query = wp.hash_grid_query(grid, xi, support_radius)
        index = int(0)

        while wp.hash_grid_query_next(query, index):
            if index == tid or (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
                continue

            world_j = particle_world[index]
            if world_i >= 0 and world_j >= 0 and world_i != world_j:
                continue

            rho_j = density[index]
            if rho_j <= 0.0:
                continue

            displacement = xi - particle_q[index]
            dist2 = wp.dot(displacement, displacement)
            if dist2 <= 0.0 or dist2 >= h2:
                continue

            grad = kernel_gradient(displacement, support_radius, kernel_family)
            laplace_weight = -wp.dot(displacement, grad) / (dist2 + eps2)
            if laplace_weight <= 0.0:
                continue

            fluid_diffusion += (particle_mass[index] / rho_j) * (particle_qd[index] - vi) * laplace_weight

    if viscosity_boundary_coefficient > 0.0 and rho_i > 0.0 and rest_density > 0.0:
        boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
        boundary_index = int(0)

        while wp.hash_grid_query_next(boundary_query, boundary_index):
            boundary_flag = boundary_flags[boundary_index]
            if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
                continue
            weight_scale = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
            if weight_scale == 0.0:
                continue

            displacement = xi - boundary_x[boundary_index]
            dist2 = wp.dot(displacement, displacement)
            if dist2 <= 0.0 or dist2 >= h2:
                continue

            grad = kernel_gradient(displacement, support_radius, kernel_family)
            laplace_weight = -wp.dot(displacement, grad) / (dist2 + eps2)
            if laplace_weight <= 0.0:
                continue

            boundary_diffusion += (
                weight_scale
                * (rest_density * boundary_volume[boundary_index] / rho_i)
                * (boundary_v[boundary_index] - vi)
                * laplace_weight
            )

    diffused_particle_qd[tid] = (
        vi
        + 10.0 * viscosity_coefficient * dt * fluid_diffusion
        + 10.0 * viscosity_boundary_coefficient * dt * boundary_diffusion
    )


@wp.kernel
def apply_viscosity_velocity_diffusion_without_grid(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    viscosity_coefficient: float,
    dt: float,
    diffused_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply viscosity diffusion for cases without a particle hash grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or viscosity_coefficient <= 0.0 or dt <= 0.0:
        diffused_particle_qd[tid] = particle_qd[tid]
        return

    vi = particle_qd[tid]
    diffused_particle_qd[tid] = vi


@wp.kernel
def apply_viscosity_velocity_diffusion_without_grid_with_boundary(
    boundary_grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    boundary_x: wp.array(dtype=wp.vec3),
    boundary_v: wp.array(dtype=wp.vec3),
    boundary_volume: wp.array(dtype=float),
    boundary_flags: wp.array(dtype=wp.int32),
    static_boundary_weight: float,
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    viscosity_boundary_coefficient: float,
    dt: float,
    diffused_particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply boundary-only viscosity diffusion for cases without a particle grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or dt <= 0.0 or viscosity_boundary_coefficient <= 0.0:
        diffused_particle_qd[tid] = particle_qd[tid]
        return

    xi = particle_q[tid]
    vi = particle_qd[tid]
    rho_i = density[tid]
    h2 = support_radius * support_radius
    eps2 = 1.0e-2 * h2
    boundary_diffusion = wp.vec3(0.0)

    if rho_i > 0.0 and rest_density > 0.0:
        boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
        boundary_index = int(0)

        while wp.hash_grid_query_next(boundary_query, boundary_index):
            boundary_flag = boundary_flags[boundary_index]
            if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
                continue
            weight_scale = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
            if weight_scale == 0.0:
                continue

            displacement = xi - boundary_x[boundary_index]
            dist2 = wp.dot(displacement, displacement)
            if dist2 <= 0.0 or dist2 >= h2:
                continue

            grad = kernel_gradient(displacement, support_radius, kernel_family)
            laplace_weight = -wp.dot(displacement, grad) / (dist2 + eps2)
            if laplace_weight <= 0.0:
                continue

            boundary_diffusion += (
                weight_scale
                * (rest_density * boundary_volume[boundary_index] / rho_i)
                * (boundary_v[boundary_index] - vi)
                * laplace_weight
            )

    diffused_particle_qd[tid] = vi + 10.0 * viscosity_boundary_coefficient * dt * boundary_diffusion


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


@wp.kernel
def apply_artificial_damping(
    x_new: wp.array(dtype=wp.vec3),
    x_star: wp.array(dtype=wp.vec3),
    x_old: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    support_radius: float,
    damping_beta: float,
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Apply the IPBF artificial damping velocity correction."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or dt <= 0.0:
        particle_qd[tid] = wp.vec3(0.0)
        return

    v = (x_new[tid] - x_old[tid]) / dt
    if damping_beta <= 0.0 or support_radius <= 0.0:
        particle_qd[tid] = v
        return

    threshold = damping_beta * support_radius
    if threshold <= 0.0:
        particle_qd[tid] = v
        return

    dx = x_new[tid] - x_star[tid]
    dx_norm = wp.length(dx)
    if dx_norm >= threshold:
        particle_qd[tid] = v
        return

    v2 = wp.dot(v, v)
    if v2 <= 0.0:
        particle_qd[tid] = v
        return

    v_star = (x_star[tid] - x_old[tid]) / dt
    v_star2 = wp.dot(v_star, v_star)
    if v_star2 >= v2:
        particle_qd[tid] = v
        return

    d = 1.0 - dx_norm / threshold
    scale2 = 1.0 - d * (v2 - v_star2) / v2
    if scale2 <= 0.0:
        particle_qd[tid] = wp.vec3(0.0)
        return

    particle_qd[tid] = wp.sqrt(scale2) * v
