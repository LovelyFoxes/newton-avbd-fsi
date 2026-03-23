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


def get_box_center(*, wall_half_height: float, floor_y: float = 0.0) -> tuple[float, float, float]:
    """Return the geometric center of an axis-aligned container box."""
    return (0.0, floor_y + wall_half_height, 0.0)


def get_particle_grid_half_span(
    *,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
) -> tuple[float, float, float]:
    """Return half the center-to-center extent of a particle grid."""
    return (
        0.5 * (dim_x - 1) * cell_x,
        0.5 * (dim_y - 1) * cell_y,
        0.5 * (dim_z - 1) * cell_z,
    )


def get_particle_grid_origin_from_center(
    *,
    center: tuple[float, float, float],
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
) -> wp.vec3:
    """Convert a desired particle-grid center to the origin expected by ``add_particle_grid``."""
    hx, hy, hz = get_particle_grid_half_span(
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
    )
    cx, cy, cz = center
    return wp.vec3(cx - hx, cy - hy, cz - hz)


def get_symmetric_wall_aligned_center_offset_x(
    *,
    container_half_width: float,
    dim_x: int,
    cell_x: float,
    gap_x: float = 0.0,
) -> float:
    """Return the symmetric x-offset that places a particle block near both side walls."""
    hx, _, _ = get_particle_grid_half_span(
        dim_x=dim_x,
        dim_y=1,
        dim_z=1,
        cell_x=cell_x,
        cell_y=1.0,
        cell_z=1.0,
    )
    return container_half_width - gap_x - hx


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
