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

"""Warp kernels for AVBD/IPBF fluid-solid boundary samples."""

import warp as wp


@wp.kernel
def accumulate_step_reaction_diagnostics(
    body_force: wp.array(dtype=wp.vec3),
    body_torque: wp.array(dtype=wp.vec3),
    body_force_step_sum: wp.array(dtype=wp.vec3),
    body_torque_step_sum: wp.array(dtype=wp.vec3),
    body_force_step_max_norm: wp.array(dtype=float),
    body_torque_step_max_norm: wp.array(dtype=float),
    body_force_step_count: wp.array(dtype=wp.int32),
):
    """Accumulate the current FSI body wrench into step-level diagnostics."""
    tid = wp.tid()

    force = body_force[tid]
    torque = body_torque[tid]
    body_force_step_sum[tid] = body_force_step_sum[tid] + force
    body_torque_step_sum[tid] = body_torque_step_sum[tid] + torque
    body_force_step_max_norm[tid] = wp.max(body_force_step_max_norm[tid], wp.length(force))
    body_torque_step_max_norm[tid] = wp.max(body_torque_step_max_norm[tid], wp.length(torque))

    if tid == 0:
        body_force_step_count[0] = body_force_step_count[0] + 1


@wp.kernel
def update_step_reaction_averages(
    body_force_step_sum: wp.array(dtype=wp.vec3),
    body_torque_step_sum: wp.array(dtype=wp.vec3),
    body_force_step_count: wp.array(dtype=wp.int32),
    body_force_step_avg: wp.array(dtype=wp.vec3),
    body_torque_step_avg: wp.array(dtype=wp.vec3),
):
    """Update step-average FSI body wrenches from accumulated sums."""
    tid = wp.tid()

    count = body_force_step_count[0]
    if count <= 0:
        body_force_step_avg[tid] = wp.vec3(0.0)
        body_torque_step_avg[tid] = wp.vec3(0.0)
        return

    inv_count = 1.0 / float(count)
    body_force_step_avg[tid] = body_force_step_sum[tid] * inv_count
    body_torque_step_avg[tid] = body_torque_step_sum[tid] * inv_count


@wp.kernel
def update_boundary_sample_world_kinematics(
    sample_body: wp.array(dtype=wp.int32),
    sample_x_local: wp.array(dtype=wp.vec3),
    sample_normal_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    sample_x_world: wp.array(dtype=wp.vec3),
    sample_v_world: wp.array(dtype=wp.vec3),
    sample_normal_world: wp.array(dtype=wp.vec3),
):
    """Update boundary sample world-space positions and velocities."""
    tid = wp.tid()
    body = sample_body[tid]
    x_local = sample_x_local[tid]
    n_local = sample_normal_local[tid]

    if body >= 0:
        X_wb = body_q[body]
        x_world = wp.transform_point(X_wb, x_local)
        n_world = wp.normalize(wp.transform_vector(X_wb, n_local))
        com_world = wp.transform_point(X_wb, body_com[body])

        body_v_s = body_qd[body]
        body_v = wp.spatial_top(body_v_s)
        body_w = wp.spatial_bottom(body_v_s)

        sample_x_world[tid] = x_world
        sample_v_world[tid] = body_v + wp.cross(body_w, x_world - com_world)
        sample_normal_world[tid] = n_world
    else:
        sample_x_world[tid] = x_local
        sample_v_world[tid] = wp.vec3(0.0)
        sample_normal_world[tid] = wp.normalize(n_local)
