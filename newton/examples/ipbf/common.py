# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the public IPBF examples."""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def scale_velocities(
    particle_qd: wp.array(dtype=wp.vec3),
    scale: float,
):
    """Apply a uniform multiplicative velocity damping."""
    tid = wp.tid()
    particle_qd[tid] = scale * particle_qd[tid]


def build_box_wireframe(
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    device,
    include_top: bool = True,
) -> tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3)]:
    """Build a box wireframe for viewer line rendering."""
    corners = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    if include_top:
        edges.extend(
            [
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ]
        )

    starts = wp.array(corners[[i for i, _ in edges]], dtype=wp.vec3, device=device)
    ends = wp.array(corners[[j for _, j in edges]], dtype=wp.vec3, device=device)
    return starts, ends


def build_ellipsoid_particle_cloud(
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    spacing: float,
    velocity: tuple[float, float, float],
    mass: float,
    radius: float,
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """Build an axis-aligned ellipsoidal particle cloud."""
    cx, cy, cz = center
    rx, ry, rz = radii
    vx, vy, vz = velocity

    px = np.arange(-rx, rx + 0.5 * spacing, spacing, dtype=np.float32)
    py = np.arange(-ry, ry + 0.5 * spacing, spacing, dtype=np.float32)
    pz = np.arange(-rz, rz + 0.5 * spacing, spacing, dtype=np.float32)
    local_points = np.stack(np.meshgrid(px, py, pz), axis=-1).reshape(-1, 3)

    normalized = (
        (local_points[:, 0] / max(rx, 1.0e-8)) ** 2
        + (local_points[:, 1] / max(ry, 1.0e-8)) ** 2
        + (local_points[:, 2] / max(rz, 1.0e-8)) ** 2
    )
    local_points = local_points[normalized <= 1.0]
    world_points = local_points + np.array([cx, cy, cz], dtype=np.float32)

    velocities = np.broadcast_to(np.array([vx, vy, vz], dtype=np.float32), world_points.shape)
    masses = [mass] * len(world_points)
    radii_out = [radius] * len(world_points)
    return world_points.tolist(), velocities.tolist(), masses, radii_out
