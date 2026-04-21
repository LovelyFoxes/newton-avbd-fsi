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
from newton.solvers import BoundarySampleFlags, FSIBoundaryModel
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    out = np.empty_like(points)
    out[:, 0] = c * points[:, 0] - s * points[:, 1]
    out[:, 1] = s * points[:, 0] + c * points[:, 1]
    out[:, 2] = points[:, 2]
    return out


def _axis_patch_widths(axis_samples: np.ndarray) -> np.ndarray:
    count = len(axis_samples)
    if count <= 1:
        return np.zeros(count, dtype=np.float32)

    widths = np.empty(count, dtype=np.float32)
    widths[0] = 0.5 * float(axis_samples[1] - axis_samples[0])
    widths[-1] = 0.5 * float(axis_samples[-1] - axis_samples[-2])
    if count > 2:
        widths[1:-1] = 0.5 * (axis_samples[2:] - axis_samples[:-2])
    return widths


def _expected_box_surface_quadrature_volumes(points: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    zs = np.unique(points[:, 2])
    x_widths = _axis_patch_widths(xs)
    y_widths = _axis_patch_widths(ys)
    z_widths = _axis_patch_widths(zs)
    x_index = {float(value): index for index, value in enumerate(xs)}
    y_index = {float(value): index for index, value in enumerate(ys)}
    z_index = {float(value): index for index, value in enumerate(zs)}

    expected = np.zeros(len(points), dtype=np.float32)
    for sample_index, point in enumerate(points):
        ix = x_index[float(point[0])]
        iy = y_index[float(point[1])]
        iz = z_index[float(point[2])]
        if np.isclose(abs(point[0]), hx, atol=1.0e-6):
            expected[sample_index] += (hx / 3.0) * y_widths[iy] * z_widths[iz]
        if np.isclose(abs(point[1]), hy, atol=1.0e-6):
            expected[sample_index] += (hy / 3.0) * x_widths[ix] * z_widths[iz]
        if np.isclose(abs(point[2]), hz, atol=1.0e-6):
            expected[sample_index] += (hz / 3.0) * x_widths[ix] * y_widths[iy]

    return expected


def _expected_box_surface_patch_areas(points: np.ndarray, hx: float, hy: float, hz: float) -> np.ndarray:
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    zs = np.unique(points[:, 2])
    x_widths = _axis_patch_widths(xs)
    y_widths = _axis_patch_widths(ys)
    z_widths = _axis_patch_widths(zs)
    x_index = {float(value): index for index, value in enumerate(xs)}
    y_index = {float(value): index for index, value in enumerate(ys)}
    z_index = {float(value): index for index, value in enumerate(zs)}

    expected = np.zeros(len(points), dtype=np.float32)
    for sample_index, point in enumerate(points):
        ix = x_index[float(point[0])]
        iy = y_index[float(point[1])]
        iz = z_index[float(point[2])]
        if np.isclose(abs(point[0]), hx, atol=1.0e-6):
            expected[sample_index] += y_widths[iy] * z_widths[iz]
        if np.isclose(abs(point[1]), hy, atol=1.0e-6):
            expected[sample_index] += x_widths[ix] * z_widths[iz]
        if np.isclose(abs(point[2]), hz, atol=1.0e-6):
            expected[sample_index] += x_widths[ix] * y_widths[iy]

    return expected


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
    sample_volume = boundary.sample_volume.numpy()
    flags = boundary.sample_flags.numpy()

    np.testing.assert_allclose(x_world, x_local, atol=1.0e-6)
    np.testing.assert_allclose(n_world, n_local, atol=1.0e-6)
    test.assertTrue(np.all(np.isfinite(sample_volume)))
    test.assertTrue(np.all(sample_volume > 0.0))
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
    sample_volume = boundary.sample_volume.numpy()

    test.assertGreater(boundary.sample_count, 0)
    test.assertTrue(np.all((flags & int(BoundarySampleFlags.STATIC)) != 0))
    test.assertTrue(np.all(np.isfinite(sample_volume)))
    test.assertTrue(np.all(sample_volume > 0.0))
    np.testing.assert_allclose(velocities, np.zeros_like(velocities), atol=1.0e-6)
    np.testing.assert_allclose(np.mean(x_world, axis=0), [1.0, 2.0, 3.0], atol=1.0e-6)


def test_dynamic_sphere_samples_follow_body(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.3)
    model = builder.finalize(device=device)
    state = model.state()

    boundary = FSIBoundaryModel(model, spacing=0.1, support_radius=0.25, device=device)
    test.assertGreater(boundary.sample_count, 12)

    boundary.update_world_kinematics(state)

    x_local = boundary.sample_x_local.numpy()
    x_world = boundary.sample_x_world.numpy()
    n_local = boundary.sample_normal_local.numpy()
    n_world = boundary.sample_normal_world.numpy()
    sample_volume = boundary.sample_volume.numpy()
    sample_area = boundary.sample_area_patch.numpy()
    quadrature_volume = boundary.sample_volume_quadrature.numpy()
    flags = boundary.sample_flags.numpy()

    np.testing.assert_allclose(x_world, x_local, atol=1.0e-6)
    np.testing.assert_allclose(n_world, n_local, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(x_local, axis=1), 0.3, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(n_local, axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(n_local, x_local / 0.3, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(sample_area), 4.0 * math.pi * 0.3 * 0.3, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        np.sum(quadrature_volume),
        (4.0 / 3.0) * math.pi * 0.3 * 0.3 * 0.3,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    test.assertTrue(np.all(np.isfinite(sample_volume)))
    test.assertTrue(np.all(sample_volume > 0.0))
    test.assertTrue(np.all((flags & int(BoundarySampleFlags.DYNAMIC)) != 0))

    angle = math.pi * 0.5
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    state.body_q.assign(
        wp.array(
            [wp.transform(wp.vec3(*translation), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle))],
            dtype=wp.transform,
            device=device,
        )
    )

    boundary.update_world_kinematics(state)

    np.testing.assert_allclose(boundary.sample_x_world.numpy(), _rotate_z(x_local, angle) + translation, atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_normal_world.numpy(), _rotate_z(n_local, angle), atol=1.0e-6)


def test_dynamic_sphere_hydrostatic_modes_match_surface_weights(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    dynamic_body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    kinematic_body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    builder.add_shape_sphere(dynamic_body, radius=0.3)
    builder.add_shape_sphere(kinematic_body, radius=0.3)
    model = builder.finalize(device=device)

    boundary = FSIBoundaryModel(
        model,
        spacing=0.1,
        support_radius=0.25,
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE,
        device=device,
    )

    sample_body = boundary.sample_body.numpy()
    raw_volume = boundary.sample_volume.numpy()
    patch_area = boundary.sample_area_patch.numpy()
    quadrature_volume = boundary.sample_volume_quadrature.numpy()
    hydrostatic_volume = boundary.sample_volume_hydrostatic.numpy()
    hydrostatic_scale = boundary.sample_volume_hydrostatic_scale.numpy()
    expected_surface_area = 4.0 * math.pi * 0.3 * 0.3
    expected_volume = (4.0 / 3.0) * math.pi * 0.3 * 0.3 * 0.3

    dynamic_mask = sample_body == dynamic_body
    kinematic_mask = sample_body == kinematic_body

    np.testing.assert_allclose(np.sum(patch_area[dynamic_mask]), expected_surface_area, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(quadrature_volume[dynamic_mask]), expected_volume, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_volume[dynamic_mask], quadrature_volume[dynamic_mask], rtol=1.0e-6)
    np.testing.assert_allclose(np.sum(hydrostatic_volume[dynamic_mask]), expected_volume, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_volume[kinematic_mask], raw_volume[kinematic_mask], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_scale[kinematic_mask], np.ones_like(hydrostatic_scale[kinematic_mask]))
    test.assertTrue(np.all(hydrostatic_scale[dynamic_mask] > 0.0))


def test_dynamic_box_hydrostatic_volume_matches_geometric_volume(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    dynamic_body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    kinematic_body = builder.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    builder.add_shape_box(dynamic_body, hx=0.5, hy=0.25, hz=0.25)
    builder.add_shape_box(kinematic_body, hx=0.5, hy=0.25, hz=0.25)
    model = builder.finalize(device=device)

    boundary = FSIBoundaryModel(
        model,
        spacing=0.25,
        support_radius=0.5,
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_VOLUME,
        device=device,
    )

    sample_body = boundary.sample_body.numpy()
    raw_volume = boundary.sample_volume.numpy()
    hydrostatic_volume = boundary.sample_volume_hydrostatic.numpy()
    hydrostatic_scale = boundary.sample_volume_hydrostatic_scale.numpy()
    expected_volume = 8.0 * 0.5 * 0.25 * 0.25

    dynamic_mask = sample_body == dynamic_body
    kinematic_mask = sample_body == kinematic_body

    test.assertTrue(np.all(dynamic_mask | kinematic_mask))
    test.assertNotAlmostEqual(float(np.sum(raw_volume[dynamic_mask])), expected_volume, places=3)
    np.testing.assert_allclose(np.sum(hydrostatic_volume[dynamic_mask]), expected_volume, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_volume[kinematic_mask], raw_volume[kinematic_mask], rtol=1.0e-6, atol=1.0e-6)
    test.assertTrue(np.all(hydrostatic_scale[dynamic_mask] > 0.0))
    np.testing.assert_allclose(hydrostatic_scale[kinematic_mask], np.ones_like(hydrostatic_scale[kinematic_mask]))


def test_dynamic_box_surface_quadrature_matches_expected_patch_weights(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    dynamic_body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    kinematic_body = builder.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    builder.add_shape_box(dynamic_body, hx=0.5, hy=0.25, hz=0.25)
    builder.add_shape_box(kinematic_body, hx=0.5, hy=0.25, hz=0.25)
    model = builder.finalize(device=device)

    boundary = FSIBoundaryModel(
        model,
        spacing=0.25,
        support_radius=0.5,
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_QUADRATURE,
        device=device,
    )

    sample_body = boundary.sample_body.numpy()
    sample_x_local = boundary.sample_x_local.numpy()
    raw_volume = boundary.sample_volume.numpy()
    patch_area = boundary.sample_area_box_patch.numpy()
    quadrature_volume = boundary.sample_volume_box_quadrature.numpy()
    hydrostatic_volume = boundary.sample_volume_hydrostatic.numpy()
    hydrostatic_scale = boundary.sample_volume_hydrostatic_scale.numpy()
    expected_volume = 8.0 * 0.5 * 0.25 * 0.25
    expected_surface_area = 2.0 * (
        (2.0 * 0.5) * (2.0 * 0.25) + (2.0 * 0.5) * (2.0 * 0.25) + (2.0 * 0.25) * (2.0 * 0.25)
    )

    dynamic_mask = sample_body == dynamic_body
    kinematic_mask = sample_body == kinematic_body
    dynamic_points = sample_x_local[dynamic_mask]
    expected_patch_area = _expected_box_surface_patch_areas(dynamic_points, hx=0.5, hy=0.25, hz=0.25)
    expected_quadrature = _expected_box_surface_quadrature_volumes(dynamic_points, hx=0.5, hy=0.25, hz=0.25)

    np.testing.assert_allclose(patch_area[dynamic_mask], expected_patch_area, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(quadrature_volume[dynamic_mask], expected_quadrature, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_volume[dynamic_mask], expected_quadrature, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(hydrostatic_volume[dynamic_mask]), expected_volume, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(patch_area[dynamic_mask]), expected_surface_area, rtol=1.0e-6, atol=1.0e-6)
    test.assertGreater(np.std(hydrostatic_scale[dynamic_mask]), 1.0e-3)
    kinematic_points = sample_x_local[kinematic_mask]
    expected_kinematic_patch_area = _expected_box_surface_patch_areas(kinematic_points, hx=0.5, hy=0.25, hz=0.25)
    expected_kinematic_quadrature = _expected_box_surface_quadrature_volumes(kinematic_points, hx=0.5, hy=0.25, hz=0.25)
    np.testing.assert_allclose(patch_area[kinematic_mask], expected_kinematic_patch_area, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        quadrature_volume[kinematic_mask],
        expected_kinematic_quadrature,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(hydrostatic_volume[kinematic_mask], raw_volume[kinematic_mask], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_scale[kinematic_mask], np.ones_like(hydrostatic_scale[kinematic_mask]))


def test_dynamic_box_surface_thickness_matches_raw_total_with_patch_distribution(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    dynamic_body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    kinematic_body = builder.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    builder.add_shape_box(dynamic_body, hx=0.5, hy=0.25, hz=0.25)
    builder.add_shape_box(kinematic_body, hx=0.5, hy=0.25, hz=0.25)
    model = builder.finalize(device=device)

    boundary = FSIBoundaryModel(
        model,
        spacing=0.25,
        support_radius=0.5,
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_THICKNESS,
        device=device,
    )

    sample_body = boundary.sample_body.numpy()
    sample_x_local = boundary.sample_x_local.numpy()
    raw_volume = boundary.sample_volume.numpy()
    patch_area = boundary.sample_area_box_patch.numpy()
    hydrostatic_volume = boundary.sample_volume_hydrostatic.numpy()
    hydrostatic_scale = boundary.sample_volume_hydrostatic_scale.numpy()

    dynamic_mask = sample_body == dynamic_body
    kinematic_mask = sample_body == kinematic_body
    dynamic_points = sample_x_local[dynamic_mask]
    expected_patch_area = _expected_box_surface_patch_areas(dynamic_points, hx=0.5, hy=0.25, hz=0.25)
    raw_total = float(np.sum(raw_volume[dynamic_mask]))
    patch_area_total = float(np.sum(expected_patch_area))
    expected_thickness = raw_total / patch_area_total
    expected_volume = expected_patch_area * expected_thickness

    np.testing.assert_allclose(patch_area[dynamic_mask], expected_patch_area, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_volume[dynamic_mask], expected_volume, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(hydrostatic_volume[dynamic_mask]), raw_total, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        hydrostatic_scale[dynamic_mask],
        np.divide(
            expected_volume,
            raw_volume[dynamic_mask],
            out=np.ones_like(expected_volume),
            where=raw_volume[dynamic_mask] > 0.0,
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    test.assertGreater(np.std(hydrostatic_scale[dynamic_mask]), 1.0e-3)
    np.testing.assert_allclose(hydrostatic_volume[kinematic_mask], raw_volume[kinematic_mask], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(hydrostatic_scale[kinematic_mask], np.ones_like(hydrostatic_scale[kinematic_mask]))


def test_step_reaction_diagnostics_accumulate_body_wrenches(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()))
    body_b = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body_a, hx=0.25, hy=0.25, hz=0.25)
    builder.add_shape_box(body_b, hx=0.25, hy=0.25, hz=0.25)
    model = builder.finalize(device=device)
    boundary = FSIBoundaryModel(model, spacing=0.25, support_radius=0.5, device=device)

    force_a = np.array([[1.0, 2.0, 2.0], [-3.0, 0.0, 4.0]], dtype=np.float32)
    torque_a = np.array([[0.0, 3.0, 4.0], [2.0, 1.0, 2.0]], dtype=np.float32)
    force_b = np.array([[0.5, -1.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    torque_b = np.array([[1.0, 0.0, 0.0], [0.0, -4.0, 3.0]], dtype=np.float32)

    boundary.body_force.assign(force_a)
    boundary.body_torque.assign(torque_a)
    boundary.accumulate_step_diagnostics()

    np.testing.assert_array_equal(boundary.body_force_step_count.numpy(), np.array([1], dtype=np.int32))
    np.testing.assert_allclose(boundary.body_force_step_sum.numpy(), force_a, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_torque_step_sum.numpy(), torque_a, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_force_step_avg.numpy(), force_a, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_torque_step_avg.numpy(), torque_a, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        boundary.body_force_step_max_norm.numpy(),
        np.linalg.norm(force_a, axis=1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    boundary.clear_forces()
    np.testing.assert_allclose(boundary.body_force.numpy(), np.zeros_like(force_a), atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_force_step_sum.numpy(), force_a, rtol=1.0e-6, atol=1.0e-6)

    boundary.body_force.assign(force_b)
    boundary.body_torque.assign(torque_b)
    boundary.accumulate_step_diagnostics()

    expected_force_sum = force_a + force_b
    expected_torque_sum = torque_a + torque_b
    np.testing.assert_array_equal(boundary.body_force_step_count.numpy(), np.array([2], dtype=np.int32))
    np.testing.assert_allclose(boundary.body_force_step_sum.numpy(), expected_force_sum, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_torque_step_sum.numpy(), expected_torque_sum, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_force_step_avg.numpy(), 0.5 * expected_force_sum, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        boundary.body_torque_step_avg.numpy(), 0.5 * expected_torque_sum, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        boundary.body_force_step_max_norm.numpy(),
        np.maximum(np.linalg.norm(force_a, axis=1), np.linalg.norm(force_b, axis=1)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    boundary.clear_step_diagnostics()
    np.testing.assert_array_equal(boundary.body_force_step_count.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(boundary.body_force_step_sum.numpy(), np.zeros_like(force_a), atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_force_step_avg.numpy(), np.zeros_like(force_a), atol=1.0e-6)
    np.testing.assert_allclose(boundary.body_force_step_max_norm.numpy(), np.zeros(model.body_count), atol=1.0e-6)


def test_triangle_samples_follow_deformable_particles(test: unittest.TestCase, device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particles(
        pos=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 2.0, 0.0),
            wp.vec3(0.0, 0.0, 3.0),
        ],
        mass=[1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05],
    )
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)
    state = model.state()

    boundary = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=2.0,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    test.assertEqual(boundary.sample_count, 1)
    test.assertEqual(boundary.triangle_sample_count, 1)
    test.assertEqual(boundary.triangle_count, 1)
    test.assertIsNotNone(boundary.triangle_contact_grid)
    test.assertIsNone(boundary.triangle_contact_bvh)

    boundary.update_world_kinematics(state)
    boundary.build_grid()

    barycentric = boundary.sample_barycentric.numpy()[0]
    q = state.particle_q.numpy().copy()
    qd = state.particle_qd.numpy().copy()
    expected_x = barycentric[0] * q[0] + barycentric[1] * q[1] + barycentric[2] * q[2]
    expected_v = barycentric[0] * qd[0] + barycentric[1] * qd[1] + barycentric[2] * qd[2]
    expected_contact_radius = float(np.linalg.norm(q - np.mean(q, axis=0), axis=1).max())
    expected_lower = np.min(q, axis=0)
    expected_upper = np.max(q, axis=0)

    np.testing.assert_allclose(barycentric, np.full(3, 1.0 / 3.0, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_x_world.numpy()[0], expected_x, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_v_world.numpy()[0], expected_v, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_normal_world.numpy()[0], [0.0, 0.0, 1.0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.triangle_contact_x_world.numpy()[0], expected_x, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.triangle_contact_aabb_lower.numpy()[0], expected_lower, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.triangle_contact_aabb_upper.numpy()[0], expected_upper, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        boundary.triangle_contact_radius.numpy(), np.array([expected_contact_radius], dtype=np.float32), atol=1.0e-6
    )
    test.assertAlmostEqual(boundary.triangle_contact_radius_max, expected_contact_radius, places=6)
    test.assertIsNotNone(boundary.triangle_contact_bvh)
    np.testing.assert_allclose(boundary.sample_area_patch.numpy(), np.array([0.5], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_volume.numpy(), np.array([0.1], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(
        boundary.sample_volume_hydrostatic.numpy(), np.array([0.1], dtype=np.float32), atol=1.0e-6
    )
    np.testing.assert_array_equal(boundary.sample_triangle.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(boundary.sample_vertex0.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(boundary.sample_vertex1.numpy(), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(boundary.sample_vertex2.numpy(), np.array([2], dtype=np.int32))
    test.assertTrue(bool((boundary.sample_flags.numpy()[0] & int(BoundarySampleFlags.DYNAMIC)) != 0))
    test.assertTrue(bool((boundary.sample_flags.numpy()[0] & int(BoundarySampleFlags.STATIC)) == 0))

    moved_q = np.array(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    moved_qd = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.particle_q.assign(moved_q)
    state.particle_qd.assign(moved_qd)
    boundary.update_world_kinematics(state)
    boundary.build_grid()

    expected_moved_x = barycentric[0] * moved_q[0] + barycentric[1] * moved_q[1] + barycentric[2] * moved_q[2]
    expected_moved_v = barycentric[0] * moved_qd[0] + barycentric[1] * moved_qd[1] + barycentric[2] * moved_qd[2]
    expected_swept_lower = np.min(np.vstack([q, moved_q]), axis=0)
    expected_swept_upper = np.max(np.vstack([q, moved_q]), axis=0)
    np.testing.assert_allclose(boundary.sample_x_world.numpy()[0], expected_moved_x, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.sample_v_world.numpy()[0], expected_moved_v, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary.triangle_contact_x_world.numpy()[0], expected_moved_x, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        boundary.triangle_contact_aabb_lower.numpy()[0], expected_swept_lower, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        boundary.triangle_contact_aabb_upper.numpy()[0], expected_swept_upper, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(boundary.sample_normal_world.numpy()[0], [0.0, 0.0, 1.0], rtol=1.0e-6, atol=1.0e-6)


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

add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_sphere_samples_follow_body",
    test_dynamic_sphere_samples_follow_body,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_sphere_hydrostatic_modes_match_surface_weights",
    test_dynamic_sphere_hydrostatic_modes_match_surface_weights,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_box_hydrostatic_volume_matches_geometric_volume",
    test_dynamic_box_hydrostatic_volume_matches_geometric_volume,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_box_surface_quadrature_matches_expected_patch_weights",
    test_dynamic_box_surface_quadrature_matches_expected_patch_weights,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_dynamic_box_surface_thickness_matches_raw_total_with_patch_distribution",
    test_dynamic_box_surface_thickness_matches_raw_total_with_patch_distribution,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_step_reaction_diagnostics_accumulate_body_wrenches",
    test_step_reaction_diagnostics_accumulate_body_wrenches,
    devices=devices,
)

add_function_test(
    TestFSIBoundaryModel,
    "test_triangle_samples_follow_deformable_particles",
    test_triangle_samples_follow_deformable_particles,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
