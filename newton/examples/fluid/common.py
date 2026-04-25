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


def register_fluid_solver_attributes(builder: newton.ModelBuilder, fluid_solver_name: str) -> None:
    """Register the state attributes required by the selected fluid solver."""
    if fluid_solver_name == "ipbf":
        SolverIPBF.register_custom_attributes(builder)
        return
    if fluid_solver_name == "pbf":
        SolverPBF.register_custom_attributes(builder)
        return
    raise ValueError(f"Unsupported fluid solver '{fluid_solver_name}'.")


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
