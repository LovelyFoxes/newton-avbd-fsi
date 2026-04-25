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

"""Shared helpers for the public fluid-solver comparison examples."""

from __future__ import annotations

from argparse import ArgumentParser

import warp as wp

import newton
from newton.solvers import FSIBoundaryModel, SolverIPBF, SolverPBF


def add_fluid_solver_argument(parser: ArgumentParser) -> ArgumentParser:
    """Add the shared ``--fluid-solver`` selector to a parser."""
    parser.add_argument(
        "--fluid-solver",
        choices=["ipbf", "pbf"],
        default="ipbf",
        help="Fluid solver used by the scene. Keeps geometry and scene parameters fixed.",
    )
    return parser


def add_shared_fluid_tuning_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add shared solver-resolution overrides used by comparison scenes."""
    parser.add_argument(
        "--fluid-iterations",
        type=int,
        default=None,
        help="Override the number of fluid density iterations while keeping the scene identical.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=None,
        help="Override the number of simulation substeps per rendered frame.",
    )
    return parser


def apply_shared_fluid_tuning_overrides(args, config: dict[str, object]) -> dict[str, object]:
    """Apply shared fluid-scene overrides in place and return ``config``."""
    fluid_iterations = getattr(args, "fluid_iterations", None)
    sim_substeps = getattr(args, "sim_substeps", None)
    if fluid_iterations is not None:
        config["iterations"] = int(fluid_iterations)
    if sim_substeps is not None:
        config["sim_substeps"] = int(sim_substeps)
    return config


def register_fluid_solver_attributes(builder: newton.ModelBuilder, fluid_solver_name: str) -> None:
    """Register the state attributes required by the selected fluid solver."""
    if fluid_solver_name == "ipbf":
        SolverIPBF.register_custom_attributes(builder)
        return
    if fluid_solver_name == "pbf":
        SolverPBF.register_custom_attributes(builder)
        return
    raise ValueError(f"Unsupported fluid solver '{fluid_solver_name}'.")


def shared_shape_contact_settings(*, particle_radius: float) -> dict[str, float]:
    """Return shared particle-shape contact settings for fluid comparison scenes.

    The values here define common scene/contact conditions shared by IPBF and
    PBF so that boundary-penetration mitigation does not bias one solver over
    the other.
    """
    radius = float(particle_radius)
    return {
        "wall_thickness": max(0.05, 6.0 * radius),
        "shape_margin": max(0.0, 0.5 * radius),
        "shape_gap": max(0.0, 2.0 * radius),
        "soft_contact_margin": max(0.05, 4.0 * radius),
    }


@wp.kernel
def clamp_particles_to_box(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    half_width_x: float,
    floor_y: float,
    top_y: float,
    half_width_z: float,
):
    """Clamp particle positions and outward normal velocity to an AABB box."""
    tid = wp.tid()

    q = particle_q[tid]
    qd = particle_qd[tid]
    radius = particle_radius[tid]

    min_x = -half_width_x + radius
    max_x = half_width_x - radius
    min_y = floor_y + radius
    max_y = top_y - radius
    min_z = -half_width_z + radius
    max_z = half_width_z - radius

    if q[0] < min_x:
        q = wp.vec3(min_x, q[1], q[2])
        if qd[0] < 0.0:
            qd = wp.vec3(0.0, qd[1], qd[2])
    elif q[0] > max_x:
        q = wp.vec3(max_x, q[1], q[2])
        if qd[0] > 0.0:
            qd = wp.vec3(0.0, qd[1], qd[2])

    if q[1] < min_y:
        q = wp.vec3(q[0], min_y, q[2])
        if qd[1] < 0.0:
            qd = wp.vec3(qd[0], 0.0, qd[2])
    elif q[1] > max_y:
        q = wp.vec3(q[0], max_y, q[2])
        if qd[1] > 0.0:
            qd = wp.vec3(qd[0], 0.0, qd[2])

    if q[2] < min_z:
        q = wp.vec3(q[0], q[1], min_z)
        if qd[2] < 0.0:
            qd = wp.vec3(qd[0], qd[1], 0.0)
    elif q[2] > max_z:
        q = wp.vec3(q[0], q[1], max_z)
        if qd[2] > 0.0:
            qd = wp.vec3(qd[0], qd[1], 0.0)

    particle_q[tid] = q
    particle_qd[tid] = qd


def create_fluid_solver(
    *,
    model: newton.Model,
    fluid_solver_name: str,
    scene: dict[str, object],
    boundary_model: FSIBoundaryModel | None = None,
):
    """Create the requested fluid solver using shared scene parameters."""
    common_kwargs = {
        "rest_density": float(scene.get("rest_density", 1000.0)),
        "smoothing_radius": float(scene["smoothing_radius"]),
        "iterations": int(scene["iterations"]),
        "viscosity_coefficient": float(scene.get("viscosity_coefficient", scene.get("viscosity", 0.0))),
        "viscosity_boundary_coefficient": float(scene.get("viscosity_boundary_coefficient", 0.0)),
        "xsph_coefficient": float(scene.get("xsph_coefficient", scene.get("xsph", 0.0))),
        "xsph_boundary_coefficient": float(scene.get("xsph_boundary_coefficient", 0.0)),
        "boundary_velocity_damping": float(scene.get("boundary_velocity_damping", 1.0)),
        "fsi_static_boundary_weight": float(scene.get("static_boundary_weight", 1.0)),
    }

    if fluid_solver_name == "ipbf":
        return SolverIPBF(
            model,
            SolverIPBF.Config(
                **common_kwargs,
                kernel_family=SolverIPBF.Config.KernelFamily.CUBIC_SPLINE,
                compliance=float(scene.get("compliance", 0.0)),
                relaxation=float(scene.get("ipbf_relaxation", 0.5)),
                use_constraint_clamp=bool(scene.get("use_constraint_clamp", True)),
            ),
            boundary_model=boundary_model,
        )

    if fluid_solver_name == "pbf":
        return SolverPBF(
            model,
            SolverPBF.Config(
                **common_kwargs,
                kernel_family=SolverPBF.Config.KernelFamily.CUBIC_SPLINE,
                relaxation=float(scene.get("pbf_relaxation", 1.0)),
                lambda_regularization=float(scene.get("lambda_regularization", 1.0e-6)),
                use_constraint_clamp=bool(scene.get("use_constraint_clamp", True)),
            ),
            boundary_model=boundary_model,
        )

    raise ValueError(f"Unsupported fluid solver '{fluid_solver_name}'.")
