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

"""PBF solver kernels."""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from ..ipbf.ipbf_kernels import (
    _BOUNDARY_SAMPLE_ACTIVE,
    body_reaction_torque_from_world_point,
    boundary_density_pressure_weight,
    equivalent_force_from_position_delta,
    kernel_gradient,
)

KERNEL_FAMILY_CUBIC_SPLINE = 0
KERNEL_FAMILY_POLY6 = 1


@wp.kernel
def mask_pbf_particle_flags(
    particle_flags: wp.array[wp.int32],
    fluid_particle_start: int,
    fluid_particle_count: int,
    pbf_particle_flags: wp.array[wp.int32],
):
    """Build solver-local particle flags for the active PBF fluid range."""
    tid = wp.tid()

    if tid < fluid_particle_start or tid >= fluid_particle_start + fluid_particle_count:
        pbf_particle_flags[tid] = 0
        return

    pbf_particle_flags[tid] = particle_flags[tid]


@wp.kernel
def initialize_constraint_and_lambda(
    density: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    rest_density: float,
    use_constraint_clamp: int,
    constraint: wp.array[wp.float32],
    lambda_value: wp.array[wp.float32],
):
    """Initialize constraint diagnostics and lambdas without a particle grid."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        constraint[tid] = 0.0
        lambda_value[tid] = 0.0
        return

    c = density[tid] / rest_density - 1.0
    if use_constraint_clamp != 0 and c < 0.0:
        c = 0.0

    constraint[tid] = c
    lambda_value[tid] = 0.0


@wp.kernel
def compute_pbf_lambda(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    density: wp.array[wp.float32],
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    use_constraint_clamp: int,
    lambda_regularization: float,
    constraint: wp.array[wp.float32],
    lambda_value: wp.array[wp.float32],
):
    """Compute classic PBF constraint values and lambda multipliers."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        constraint[tid] = 0.0
        lambda_value[tid] = 0.0
        return

    c = density[tid] / rest_density - 1.0
    if use_constraint_clamp != 0 and c < 0.0:
        constraint[tid] = 0.0
        lambda_value[tid] = 0.0
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    grad_i = wp.vec3(0.0)
    sum_grad_sq = float(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue
        if index == tid:
            continue

        displacement = xi - particle_q[index]
        grad = (particle_mass[index] / rest_density) * kernel_gradient(displacement, support_radius, kernel_family)
        grad_i = grad_i + grad
        sum_grad_sq = sum_grad_sq + wp.dot(grad, grad)

    sum_grad_sq = sum_grad_sq + wp.dot(grad_i, grad_i)
    constraint[tid] = c
    lambda_value[tid] = -c / (sum_grad_sq + lambda_regularization)


@wp.kernel
def compute_pbf_lambda_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    boundary_x: wp.array[wp.vec3],
    boundary_triangle: wp.array[wp.int32],
    boundary_volume: wp.array[wp.float32],
    boundary_flags: wp.array[wp.int32],
    static_boundary_weight: float,
    density: wp.array[wp.float32],
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    use_constraint_clamp: int,
    lambda_regularization: float,
    constraint: wp.array[wp.float32],
    lambda_value: wp.array[wp.float32],
):
    """Compute PBF lambdas with rigid boundary-sample density support."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        constraint[tid] = 0.0
        lambda_value[tid] = 0.0
        return

    c = density[tid] / rest_density - 1.0
    if use_constraint_clamp != 0 and c < 0.0:
        constraint[tid] = 0.0
        lambda_value[tid] = 0.0
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    grad_i = wp.vec3(0.0)
    sum_grad_sq = float(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue
        if index == tid:
            continue

        displacement = xi - particle_q[index]
        grad = (particle_mass[index] / rest_density) * kernel_gradient(displacement, support_radius, kernel_family)
        grad_i = grad_i + grad
        sum_grad_sq = sum_grad_sq + wp.dot(grad, grad)

    boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    boundary_index = int(0)

    while wp.hash_grid_query_next(boundary_query, boundary_index):
        boundary_flag = boundary_flags[boundary_index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        if boundary_triangle[boundary_index] >= 0:
            continue

        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        displacement = xi - boundary_x[boundary_index]
        grad = weight * boundary_volume[boundary_index] * kernel_gradient(displacement, support_radius, kernel_family)
        grad_i = grad_i + grad

    sum_grad_sq = sum_grad_sq + wp.dot(grad_i, grad_i)
    constraint[tid] = c
    lambda_value[tid] = -c / (sum_grad_sq + lambda_regularization)


@wp.func
def tensile_correction(
    displacement: wp.vec3,
    support_radius: float,
    kernel_family: int,
    tensile_instability_scale: float,
    tensile_instability_distance: float,
    tensile_instability_power: float,
) -> float:
    """Return the optional PBF tensile instability correction."""
    if tensile_instability_scale == 0.0 or tensile_instability_distance <= 0.0:
        return 0.0

    numerator = kernel_gradient(displacement, support_radius, kernel_family)
    if wp.dot(numerator, numerator) == 0.0:
        return 0.0

    w = wp.length(numerator)
    ref_disp = wp.vec3(tensile_instability_distance, 0.0, 0.0)
    w_ref = wp.length(kernel_gradient(ref_disp, support_radius, kernel_family))
    if w_ref <= 0.0:
        return 0.0

    return -tensile_instability_scale * wp.pow(w / w_ref, tensile_instability_power)


@wp.kernel
def compute_pbf_delta(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    lambda_value: wp.array[wp.float32],
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    tensile_instability_scale: float,
    tensile_instability_distance: float,
    tensile_instability_power: float,
    delta_q: wp.array[wp.vec3],
):
    """Compute PBF position corrections from neighboring fluid particles."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        delta_q[tid] = wp.vec3(0.0)
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    lambda_i = lambda_value[tid]
    delta = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue
        if index == tid:
            continue

        displacement = xi - particle_q[index]
        s_corr = tensile_correction(
            displacement,
            support_radius,
            kernel_family,
            tensile_instability_scale,
            tensile_instability_distance,
            tensile_instability_power,
        )
        grad = particle_mass[index] * kernel_gradient(displacement, support_radius, kernel_family)
        delta = delta + (lambda_i + lambda_value[index] + s_corr) * grad
    delta_q[tid] = delta / rest_density


@wp.kernel
def compute_pbf_delta_with_boundary(
    grid: wp.uint64,
    boundary_grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    lambda_value: wp.array[wp.float32],
    boundary_x: wp.array[wp.vec3],
    boundary_body: wp.array[wp.int32],
    boundary_triangle: wp.array[wp.int32],
    boundary_volume: wp.array[wp.float32],
    boundary_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rest_density: float,
    support_radius: float,
    kernel_family: int,
    static_boundary_weight: float,
    tensile_instability_scale: float,
    tensile_instability_distance: float,
    tensile_instability_power: float,
    dt: float,
    solve_relaxation: float,
    reaction_relaxation: float,
    delta_q: wp.array[wp.vec3],
    sample_force: wp.array[wp.vec3],
    body_force: wp.array[wp.vec3],
    body_torque: wp.array[wp.vec3],
):
    """Compute PBF position corrections and rigid reaction from boundary terms."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or rest_density <= 0.0:
        delta_q[tid] = wp.vec3(0.0)
        return

    xi = particle_q[tid]
    world_i = particle_world[tid]
    lambda_i = lambda_value[tid]
    delta = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, xi, support_radius)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue

        world_j = particle_world[index]
        if world_i >= 0 and world_j >= 0 and world_i != world_j:
            continue
        if index == tid:
            continue

        displacement = xi - particle_q[index]
        s_corr = tensile_correction(
            displacement,
            support_radius,
            kernel_family,
            tensile_instability_scale,
            tensile_instability_distance,
            tensile_instability_power,
        )
        grad = particle_mass[index] * kernel_gradient(displacement, support_radius, kernel_family)
        delta = delta + (lambda_i + lambda_value[index] + s_corr) * grad

    boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    boundary_index = int(0)

    while wp.hash_grid_query_next(boundary_query, boundary_index):
        boundary_flag = boundary_flags[boundary_index]
        if (boundary_flag & _BOUNDARY_SAMPLE_ACTIVE) == 0:
            continue
        if boundary_triangle[boundary_index] >= 0:
            continue

        weight = boundary_density_pressure_weight(boundary_flag, static_boundary_weight)
        if weight == 0.0:
            continue

        displacement = xi - boundary_x[boundary_index]
        grad = weight * boundary_volume[boundary_index] * kernel_gradient(displacement, support_radius, kernel_family)
        if wp.dot(grad, grad) == 0.0:
            continue

        boundary_delta = lambda_i * grad
        delta = delta + boundary_delta

        if dt > 0.0 and reaction_relaxation != 0.0 and solve_relaxation != 0.0:
            actual_delta = solve_relaxation * (boundary_delta / rest_density)
            force_on_boundary = equivalent_force_from_position_delta(
                actual_delta,
                particle_mass[tid],
                dt,
                reaction_relaxation,
            )
            if wp.dot(force_on_boundary, force_on_boundary) > 0.0:
                wp.atomic_add(sample_force, boundary_index, force_on_boundary)

                body_index = boundary_body[boundary_index]
                if body_index >= 0:
                    X_wb = body_q[body_index]
                    torque = body_reaction_torque_from_world_point(
                        boundary_x[boundary_index],
                        force_on_boundary,
                        X_wb,
                        body_com[body_index],
                    )
                    wp.atomic_add(body_force, body_index, force_on_boundary)
                    wp.atomic_add(body_torque, body_index, torque)

    delta_q[tid] = delta / rest_density
