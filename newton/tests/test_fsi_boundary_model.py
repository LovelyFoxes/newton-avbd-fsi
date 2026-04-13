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

"""Tests for the AVBD/IPBF FSI boundary sample model."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fsi import BoundarySampleFlags, FSIBoundaryModel
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    out = np.empty_like(points)
    out[:, 0] = c * points[:, 0] - s * points[:, 1]
    out[:, 1] = s * points[:, 0] + c * points[:, 1]
    out[:, 2] = points[:, 2]
    return out


def test_dynamic_box_samples_follow_body(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.5, hy=0.25, hz=0.25)
    model = builder.finalize(device=device)
    state = model.state()

    boundary = FSIBoundaryModel(model, spacing=0.25, support_radius=0.5, device=device)
    test.assertGreater(boundary.sample_count, 0)

    boundary.update_world_kinematics(state)

    x_local = boundary.sample_x_local.numpy()
    x_world = boundary.sample_x_world.numpy()
    n_local = boundary.sample_normal_local.numpy()
    n_world = boundary.sample_normal_world.numpy()
    flags = boundary.sample_flags.numpy()

    np.testing.assert_allclose(x_world, x_local, atol=1.0e-6)
    np.testing.assert_allclose(n_world, n_local, atol=1.0e-6)
    test.assertTrue(np.all((flags & int(BoundarySampleFlags.DYNAMIC)) != 0))

    extents = np.max(np.abs(x_local), axis=0)
    np.testing.assert_allclose(extents, [0.5, 0.25, 0.25], atol=1.0e-6)
    on_surface = np.isclose(np.abs(x_local), np.array([0.5, 0.25, 0.25]), atol=1.0e-6).any(axis=1)
    test.assertTrue(bool(np.all(on_surface)))

    angle = math.pi * 0.5
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    linear_velocity = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    angular_velocity = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    state.body_q.assign(
        wp.array(
            [wp.transform(wp.vec3(*translation), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle))],
            dtype=wp.transform,
            device=device,
        )
    )
    state.body_qd.assign(np.array([[*linear_velocity, *angular_velocity]], dtype=np.float32))

    boundary.update_world_kinematics(state)

    x_world = boundary.sample_x_world.numpy()
    v_world = boundary.sample_v_world.numpy()
    n_world = boundary.sample_normal_world.numpy()

    expected_x = _rotate_z(x_local, angle) + translation
    expected_n = _rotate_z(n_local, angle)
    com_local = model.body_com.numpy()[0]
    com_world = _rotate_z(com_local.reshape(1, 3), angle)[0] + translation
    expected_v = linear_velocity + np.cross(angular_velocity, expected_x - com_world)

    np.testing.assert_allclose(x_world, expected_x, atol=1.0e-6)
    np.testing.assert_allclose(n_world, expected_n, atol=1.0e-6)
    np.testing.assert_allclose(v_world, expected_v, atol=1.0e-6)


def test_static_box_samples_do_not_require_bodies(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()),
        hx=0.25,
        hy=0.25,
        hz=0.25,
    )
    model = builder.finalize(device=device)
    state = model.state()

    boundary = FSIBoundaryModel(model, spacing=0.25, support_radius=0.5, device=device)
    boundary.update_world_kinematics(state)

    flags = boundary.sample_flags.numpy()
    velocities = boundary.sample_v_world.numpy()
    x_world = boundary.sample_x_world.numpy()

    test.assertGreater(boundary.sample_count, 0)
    test.assertTrue(np.all((flags & int(BoundarySampleFlags.STATIC)) != 0))
    np.testing.assert_allclose(velocities, np.zeros_like(velocities), atol=1.0e-6)
    np.testing.assert_allclose(np.mean(x_world, axis=0), [1.0, 2.0, 3.0], atol=1.0e-6)


devices = get_test_devices(mode="basic")


class TestFSIBoundaryModel(unittest.TestCase):
    pass


add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_box_samples_follow_body",
    test_dynamic_box_samples_follow_body,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_static_box_samples_do_not_require_bodies",
    test_static_box_samples_do_not_require_bodies,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
