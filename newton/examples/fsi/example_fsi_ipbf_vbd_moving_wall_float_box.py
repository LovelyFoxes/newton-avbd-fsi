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

###########################################################################
# Example FSI IPBF VBD Moving Wall Float Box
#
# Open-top AVBD/IPBF fluid-solid interaction scene with a kinematic moving
# wall that pushes a water block toward a floating AVBD box. The right wall
# performs a continuous smooth oscillation after a short startup delay while the
# light box begins slightly above the free surface so it can settle into a
# buoyant state without starting in penetration.
#
# Command: python -m newton.examples fsi_ipbf_vbd_moving_wall_float_box
#
###########################################################################

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ipbf.common import (
    build_box_wireframe,
    get_particle_grid_half_span,
    get_particle_grid_origin_from_center,
    scale_velocities,
)
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD


class Example:
    """Moving-wall pool scene with a floating AVBD box."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used by the moving-wall float-box scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of IPBF/AVBD iteration pairs used per simulation substep.",
        )
        parser.add_argument(
            "--ipbf-iterations",
            type=int,
            default=None,
            help="Override the number of IPBF density iterations used in the scene.",
        )
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=None,
            help="Override the number of simulation substeps per rendered frame.",
        )
        parser.add_argument(
            "--projection-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to particle-shape projection reaction forces.",
        )
        parser.add_argument(
            "--velocity-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to boundary velocity projection reaction forces.",
        )
        parser.add_argument(
            "--pressure-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to pressure-gradient reaction forces.",
        )
        parser.add_argument(
            "--wall-travel",
            type=float,
            default=None,
            help="Override the oscillation travel of the moving wall [m].",
        )
        parser.add_argument(
            "--wall-frequency",
            type=float,
            default=None,
            help="Override the oscillation frequency of the moving wall [Hz].",
        )
        parser.add_argument(
            "--wall-start-delay",
            type=float,
            default=None,
            help="Override the delay before the moving wall starts oscillating [s].",
        )
        parser.add_argument(
            "--wall-motion",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable kinematic moving-wall motion.",
        )
        parser.add_argument(
            "--box-offset-x",
            type=float,
            default=None,
            help="Override the initial floating-box center offset along x [m].",
        )
        parser.add_argument(
            "--box-bottom-gap",
            type=float,
            default=None,
            help="Override the initial gap between the water surface and the box bottom [m].",
        )
        parser.add_argument(
            "--box-density",
            type=float,
            default=None,
            help="Override the floating-box density [kg/m^3].",
        )
        parser.add_argument(
            "--enable-runtime-diagnostics",
            action="store_true",
            help="Enable per-frame host-side diagnostics in interactive runs.",
        )
        parser.add_argument(
            "--include-static-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Include sampled static walls in the FSI boundary density and local dissipation paths.",
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--static-boundary-weight",
            type=float,
            default=None,
            help="Diagnostic weight applied only to static boundary-sample density/pressure contributions.",
        )
        parser.add_argument(
            "--viscosity-boundary-coefficient",
            type=float,
            default=None,
            help="Boundary-sample viscosity coefficient used for local wall/body dissipation.",
        )
        parser.add_argument(
            "--viscosity-coefficient",
            type=float,
            default=None,
            help="Fluid-fluid viscosity coefficient used for internal velocity diffusion.",
        )
        parser.add_argument(
            "--xsph-boundary-coefficient",
            type=float,
            default=None,
            help="Boundary-sample XSPH coefficient used for local wall/body velocity smoothing.",
        )
        parser.add_argument(
            "--xsph-coefficient",
            type=float,
            default=None,
            help="Fluid-fluid XSPH velocity smoothing coefficient.",
        )
        parser.add_argument(
            "--boundary-velocity-damping",
            type=float,
            default=None,
            help="Tangential velocity damping applied at final particle-shape contacts.",
        )
        parser.add_argument(
            "--velocity-damping",
            type=float,
            default=None,
            help="Example-level global velocity damping multiplier.",
        )
        parser.add_argument(
            "--hydrostatic-volume-mode",
            choices=[
                "raw",
                "dynamic-box-volume",
                "dynamic-box-surface-quadrature",
                "dynamic-box-surface-thickness",
            ],
            default="dynamic-box-surface-thickness",
            help="Hydrostatic boundary-volume model used for dynamic box boundary samples.",
        )
        return parser

    @staticmethod
    def _parse_hydrostatic_volume_mode(mode: str) -> FSIBoundaryModel.HydrostaticVolumeMode:
        mode_map = {
            "raw": FSIBoundaryModel.HydrostaticVolumeMode.NONE,
            "dynamic-box-volume": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_VOLUME,
            "dynamic-box-surface-quadrature": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_QUADRATURE,
            "dynamic-box-surface-thickness": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_SURFACE_THICKNESS,
        }
        return mode_map[str(mode)]

    def _get_scene_config(self) -> dict[str, object]:
        projection_relaxation = getattr(self.args, "projection_reaction_relaxation", None)
        velocity_reaction_relaxation = getattr(self.args, "velocity_reaction_relaxation", None)
        pressure_relaxation = getattr(self.args, "pressure_reaction_relaxation", None)
        ipbf_iterations_override = getattr(self.args, "ipbf_iterations", None)
        sim_substeps_override = getattr(self.args, "sim_substeps", None)
        wall_travel_override = getattr(self.args, "wall_travel", None)
        wall_frequency_override = getattr(self.args, "wall_frequency", None)
        wall_start_delay_override = getattr(self.args, "wall_start_delay", None)
        box_offset_x_override = getattr(self.args, "box_offset_x", None)
        box_bottom_gap_override = getattr(self.args, "box_bottom_gap", None)
        box_density_override = getattr(self.args, "box_density", None)
        include_static_override = getattr(self.args, "include_static_boundary_samples", None)
        static_boundary_weight_override = getattr(self.args, "static_boundary_weight", None)

        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.42,
                "container_half_depth": 0.34,
                "wall_half_height": 0.66,
                "pool_dim_x": 16,
                "pool_dim_y": 8,
                "pool_dim_z": 12,
                "cell": 0.04,
                "mass": 0.064,
                "radius_mean": 0.016,
                "smoothing_radius": 0.075,
                "pool_bottom_clearance": 0.03,
                "box_half_extent": wp.vec3(0.07, 0.045, 0.07),
                "box_offset_x": -0.06,
                "box_bottom_gap": 0.02,
                "box_density": 180.0,
                "rest_density": 1000.0,
                "ipbf_iterations": 6,
                "ipbf_relaxation": 0.5,
                "compliance": 1.0e-5,
                "sim_substeps": 6,
                "velocity_damping": 0.996,
                "viscosity_coefficient": 0.002,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.004,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "rigid_iterations": 2,
                "boundary_spacing": 0.04,
                "wall_travel": 0.06,
                "wall_frequency": 0.65,
                "wall_start_delay": 0.45,
                "projection_reaction_relaxation": (
                    0.0 if projection_relaxation is None else float(projection_relaxation)
                ),
                "velocity_reaction_relaxation": (
                    0.0 if velocity_reaction_relaxation is None else float(velocity_reaction_relaxation)
                ),
                "pressure_reaction_relaxation": (1.50 if pressure_relaxation is None else float(pressure_relaxation)),
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "expected_min_box_shift": 0.01,
                "expected_min_box_force_norm": 0.50,
                "expected_min_sample_force_norm": 0.02,
            }
        else:
            config = {
                "container_half_width": 0.60,
                "container_half_depth": 0.42,
                "wall_half_height": 0.92,
                "pool_dim_x": 60,
                "pool_dim_y": 16,
                "pool_dim_z": 48,
                "cell": 0.014,
                "mass": 0.002744,
                "radius_mean": 0.006,
                "smoothing_radius": 0.025,
                "pool_bottom_clearance": 0.03,
                "box_half_extent": wp.vec3(0.09, 0.06, 0.09),
                "box_offset_x": -0.10,
                "box_bottom_gap": 0.03,
                "box_density": 100.0,
                "rest_density": 1000.0,
                "ipbf_iterations": 2,
                "ipbf_relaxation": 0.5,
                "compliance": 1.0e-5,
                "sim_substeps": 6,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.005,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "rigid_iterations": 2,
                "boundary_spacing": 0.014,
                "wall_travel": 0.12,
                "wall_frequency": 0.55,
                "wall_start_delay": 0.75,
                "projection_reaction_relaxation": (
                    0.0 if projection_relaxation is None else float(projection_relaxation)
                ),
                "velocity_reaction_relaxation": (
                    0.0 if velocity_reaction_relaxation is None else float(velocity_reaction_relaxation)
                ),
                "pressure_reaction_relaxation": (1.50 if pressure_relaxation is None else float(pressure_relaxation)),
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "expected_min_box_shift": 0.0,
                "expected_min_box_force_norm": 0.0,
                "expected_min_sample_force_norm": 0.0,
            }

        if ipbf_iterations_override is not None:
            config["ipbf_iterations"] = int(ipbf_iterations_override)
        if sim_substeps_override is not None:
            config["sim_substeps"] = int(sim_substeps_override)
        if wall_travel_override is not None:
            config["wall_travel"] = float(wall_travel_override)
        if wall_frequency_override is not None:
            config["wall_frequency"] = float(wall_frequency_override)
        if wall_start_delay_override is not None:
            config["wall_start_delay"] = float(wall_start_delay_override)
        if box_offset_x_override is not None:
            config["box_offset_x"] = float(box_offset_x_override)
        if box_bottom_gap_override is not None:
            config["box_bottom_gap"] = float(box_bottom_gap_override)
        if box_density_override is not None:
            config["box_density"] = float(box_density_override)
        if include_static_override is not None:
            config["include_static_boundary_samples"] = bool(include_static_override)
        if static_boundary_weight_override is not None:
            config["static_boundary_weight"] = float(static_boundary_weight_override)

        optional_float_overrides = {
            "velocity_damping": getattr(self.args, "velocity_damping", None),
            "viscosity_coefficient": getattr(self.args, "viscosity_coefficient", None),
            "viscosity_boundary_coefficient": getattr(self.args, "viscosity_boundary_coefficient", None),
            "xsph_coefficient": getattr(self.args, "xsph_coefficient", None),
            "xsph_boundary_coefficient": getattr(self.args, "xsph_boundary_coefficient", None),
            "boundary_velocity_damping": getattr(self.args, "boundary_velocity_damping", None),
        }
        for key, value in optional_float_overrides.items():
            if value is not None:
                config[key] = float(value)

        return config

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False
        self.wall_motion_enabled = bool(getattr(self.args, "wall_motion", True))
        self.hydrostatic_volume_mode = self._parse_hydrostatic_volume_mode(
            getattr(self.args, "hydrostatic_volume_mode", "dynamic-box-surface-thickness")
        )
        self.config = self._get_scene_config()
        self.include_static_boundary_samples = bool(self.config["include_static_boundary_samples"])
        self.show_boundary_samples = bool(getattr(self.args, "show_boundary_samples", False))
        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(self.config["container_half_width"])
        self.container_half_depth = float(self.config["container_half_depth"])
        self.wall_half_height = float(self.config["wall_half_height"])
        self.wall_thickness = 0.05
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(self.config["velocity_damping"])
        self.box_half_extent = self.config["box_half_extent"]
        self.wall_travel = float(self.config["wall_travel"])
        self.wall_frequency = float(self.config["wall_frequency"])
        self.wall_start_delay = float(self.config["wall_start_delay"])
        self.current_right_wall_center_x = self.container_half_width + self.wall_thickness

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0

        self._add_container(builder)
        self.float_box_body = self._add_float_box(builder)
        self._add_pool(builder)
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            hydrostatic_volume_mode=self.hydrostatic_volume_mode,
            include_static=self.include_static_boundary_samples,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=float(self.config["rest_density"]),
                smoothing_radius=float(self.config["smoothing_radius"]),
                compliance=float(self.config["compliance"]),
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=float(self.config["ipbf_relaxation"]),
                viscosity_coefficient=float(self.config["viscosity_coefficient"]),
                viscosity_boundary_coefficient=float(self.config["viscosity_boundary_coefficient"]),
                xsph_coefficient=float(self.config["xsph_coefficient"]),
                xsph_boundary_coefficient=float(self.config["xsph_boundary_coefficient"]),
                boundary_velocity_damping=float(self.config["boundary_velocity_damping"]),
                fsi_projection_reaction_relaxation=float(self.config["projection_reaction_relaxation"]),
                fsi_velocity_projection_reaction_relaxation=float(self.config["velocity_reaction_relaxation"]),
                fsi_pressure_reaction_relaxation=float(self.config["pressure_reaction_relaxation"]),
                fsi_static_boundary_weight=float(self.config["static_boundary_weight"]),
            ),
            boundary_model=self.boundary_model,
        )
        self.solid_solver = SolverVBD(
            self.model,
            iterations=int(self.config["rigid_iterations"]),
            integrate_particles=False,
            fsi_boundary_model=self.boundary_model,
        )
        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
            config=SolverFSI.Config(
                mode=(
                    SolverFSI.Config.CouplingMode.INTERLINKED
                    if getattr(self.args, "coupling_mode", "interlinked") == "interlinked"
                    else SolverFSI.Config.CouplingMode.LOOSE
                ),
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 3))),
            ),
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=float(self.config["radius_mean"]) * 2.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.12, 0.58, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * 0.16,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.boundary_colors = self._build_boundary_colors()
        self.float_box_color = wp.array([wp.vec3(1.0, 0.68, 0.24)], dtype=wp.vec3, device=self.model.device)
        self.moving_wall_color = wp.array([wp.vec3(0.82, 0.86, 0.90)], dtype=wp.vec3, device=self.model.device)
        self.rigid_material = wp.array([wp.vec4(0.45, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.enable_runtime_diagnostics = bool(getattr(self.args, "enable_runtime_diagnostics", False)) or bool(
            getattr(self.args, "test", False)
        )

        self.container_wire_color = (0.88, 0.90, 0.95)
        self.pool_wire_starts = None
        self.pool_wire_ends = None
        self.initial_box_x = 0.0
        self.min_box_x = 0.0
        self.max_box_x = 0.0
        self.initial_surface_y = 0.0
        self.max_density = 0.0
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_box_linear_speed = 0.0
        self.max_box_angular_speed = 0.0
        self.max_box_force_norm = 0.0
        self.max_box_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.states_remain_finite = True

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(2.10, 1.55, 2.10),
            pitch=-23.0,
            yaw=-129.0,
        )

        self.reset()
        self.capture()

    def _build_boundary_colors(self) -> wp.array(dtype=wp.vec3):
        sample_body = self.boundary_model.sample_body.numpy()
        colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        colors[sample_body == self.float_box_body] = np.array([1.0, 0.68, 0.24], dtype=np.float32)
        colors[sample_body == self.moving_wall_body] = np.array([0.92, 0.94, 0.98], dtype=np.float32)
        return wp.array(colors, dtype=wp.vec3, device=self.model.device)

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

    def _add_container(self, builder: newton.ModelBuilder) -> None:
        wall_t = self.wall_thickness
        hx = self.container_half_width
        hz = self.container_half_depth
        hy = self.wall_half_height
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, -wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-(hx + wall_t), hy, 0.0), wp.quat_identity()),
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, hz + wall_t), wp.quat_identity()),
            hx=hx,
            hy=hy + wall_t,
            hz=wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, -(hz + wall_t)), wp.quat_identity()),
            hx=hx,
            hy=hy + wall_t,
            hz=wall_t,
            cfg=wall_cfg,
        )

        self.moving_wall_body = builder.add_body(
            xform=wp.transform((hx + wall_t, hy, 0.0), wp.quat_identity()),
            is_kinematic=True,
            label="fsi_moving_wall",
        )
        builder.add_shape_box(
            body=self.moving_wall_body,
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )

    def _compute_pool_surface_y(self) -> float:
        _, pool_half_span_y, _ = get_particle_grid_half_span(
            dim_x=int(self.config["pool_dim_x"]),
            dim_y=int(self.config["pool_dim_y"]),
            dim_z=int(self.config["pool_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
        )
        return (
            self.floor_y
            + float(self.config["pool_bottom_clearance"])
            + 2.0 * pool_half_span_y
            + float(self.config["radius_mean"])
        )

    def _add_float_box(self, builder: newton.ModelBuilder) -> int:
        water_surface_y = self._compute_pool_surface_y()
        hy = float(self.box_half_extent[1])
        box_center_y = water_surface_y + float(self.config["box_bottom_gap"]) + hy
        box_center = wp.vec3(float(self.config["box_offset_x"]), box_center_y, 0.0)

        box_body = builder.add_body(
            xform=wp.transform(box_center, wp.quat_identity()),
            label="fsi_float_box",
        )
        builder.add_shape_box(
            body=box_body,
            hx=float(self.box_half_extent[0]),
            hy=hy,
            hz=float(self.box_half_extent[2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=float(self.config["box_density"]),
                mu=0.0,
            ),
        )
        return box_body

    def _add_pool(self, builder: newton.ModelBuilder) -> None:
        pool_center = (
            0.0,
            self.floor_y
            + float(self.config["pool_bottom_clearance"])
            + get_particle_grid_half_span(
                dim_x=int(self.config["pool_dim_x"]),
                dim_y=int(self.config["pool_dim_y"]),
                dim_z=int(self.config["pool_dim_z"]),
                cell_x=float(self.config["cell"]),
                cell_y=float(self.config["cell"]),
                cell_z=float(self.config["cell"]),
            )[1],
            0.0,
        )
        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=pool_center,
                dim_x=int(self.config["pool_dim_x"]),
                dim_y=int(self.config["pool_dim_y"]),
                dim_z=int(self.config["pool_dim_z"]),
                cell_x=float(self.config["cell"]),
                cell_y=float(self.config["cell"]),
                cell_z=float(self.config["cell"]),
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(self.config["pool_dim_x"]),
            dim_y=int(self.config["pool_dim_y"]),
            dim_z=int(self.config["pool_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
            mass=float(self.config["mass"]),
            jitter=0.0,
            radius_mean=float(self.config["radius_mean"]),
        )

    def _moving_wall_state(self, time_s: float) -> tuple[float, float]:
        start_x = self.container_half_width + self.wall_thickness
        if not self.wall_motion_enabled or self.wall_travel == 0.0 or self.wall_frequency == 0.0:
            return start_x, 0.0
        if time_s <= self.wall_start_delay:
            return start_x, 0.0
        omega = 2.0 * math.pi * self.wall_frequency
        shifted_time = time_s - self.wall_start_delay
        center_x = start_x - 0.5 * self.wall_travel * (1.0 - math.cos(omega * shifted_time))
        velocity_x = -0.5 * self.wall_travel * omega * math.sin(omega * shifted_time)
        return center_x, velocity_x

    def _set_moving_wall_state(self, state: newton.State, time_s: float) -> None:
        center_x, velocity_x = self._moving_wall_state(time_s)
        self.current_right_wall_center_x = center_x

        body_q_np = state.body_q.numpy()
        body_q_np[self.moving_wall_body][0] = center_x
        body_q_np[self.moving_wall_body][1] = self.wall_half_height
        body_q_np[self.moving_wall_body][2] = 0.0
        body_q_np[self.moving_wall_body][3] = 0.0
        body_q_np[self.moving_wall_body][4] = 0.0
        body_q_np[self.moving_wall_body][5] = 0.0
        body_q_np[self.moving_wall_body][6] = 1.0
        state.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)

        body_qd_np = state.body_qd.numpy()
        body_qd_np[self.moving_wall_body][0] = velocity_x
        body_qd_np[self.moving_wall_body][1] = 0.0
        body_qd_np[self.moving_wall_body][2] = 0.0
        body_qd_np[self.moving_wall_body][3] = 0.0
        body_qd_np[self.moving_wall_body][4] = 0.0
        body_qd_np[self.moving_wall_body][5] = 0.0
        state.body_qd = wp.array(body_qd_np, dtype=wp.spatial_vector, device=self.model.device)

    def _update_pool_wireframe(self) -> None:
        right_interior_x = self.current_right_wall_center_x - self.wall_thickness
        self.pool_wire_starts, self.pool_wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=right_interior_x,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
            include_top=False,
        )

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Wall motion: {'on' if self.wall_motion_enabled else 'off'}")
        ui.text(f"Static boundary samples: {'on' if self.include_static_boundary_samples else 'off'}")
        ui.text(f"Render boundary samples: {'on' if self.show_boundary_samples else 'off'}")
        ui.text(f"Static boundary weight: {float(self.config['static_boundary_weight']):.2f}")
        ui.text(f"Hydrostatic mode: {self.args.hydrostatic_volume_mode!s}")
        ui.text(f"Wall travel: {self.wall_travel:.3f} m")
        ui.text(f"Wall frequency: {self.wall_frequency:.2f} Hz")
        ui.text(f"Wall start delay: {self.wall_start_delay:.2f} s")
        ui.text(f"Box density: {float(self.config['box_density']):.1f} kg/m^3")

    def reset(self):
        self.sim_time = 0.0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self._set_moving_wall_state(self.state_0, 0.0)
        self._set_moving_wall_state(self.state_1, 0.0)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()
        self._update_pool_wireframe()

        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        self.initial_box_x = float(body_q[self.float_box_body, 0])
        self.min_box_x = self.initial_box_x
        self.max_box_x = self.initial_box_x
        self.initial_surface_y = float(np.max(particle_q[:, 1] + particle_radius))
        self.max_density = 0.0
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_box_linear_speed = 0.0
        self.max_box_angular_speed = 0.0
        self.max_box_force_norm = 0.0
        self.max_box_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.states_remain_finite = True
        self.viewer._paused = True

    def capture(self):
        # Keep graph capture opt-in only: the kinematic piston is updated from
        # Python every substep, so direct launches are the most reliable path.
        self.graph = None

    def _record_diagnostics(self) -> None:
        density = self.state_0.ipbf.density.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_force = self.boundary_model.body_force.numpy()
        body_force_step_avg = self.boundary_model.body_force_step_avg.numpy()
        sample_force = self.boundary_model.sample_force.numpy()
        sample_body = self.boundary_model.sample_body.numpy()

        box_x = float(body_q[self.float_box_body, 0])
        self.min_box_x = min(self.min_box_x, box_x)
        self.max_box_x = max(self.max_box_x, box_x)
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(density).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
            and np.isfinite(body_q).all()
            and np.isfinite(body_qd).all()
            and np.isfinite(body_force).all()
            and np.isfinite(body_force_step_avg).all()
            and np.isfinite(sample_force).all()
        )

        if density.size:
            self.max_density = max(self.max_density, float(np.max(density)))
            self.max_density_ratio = max(
                self.max_density_ratio,
                float(np.max(density) / max(float(self.config["rest_density"]), 1.0e-8)),
            )
        if particle_qd.size:
            self.max_particle_speed = max(self.max_particle_speed, float(np.linalg.norm(particle_qd, axis=1).max()))
        if body_qd.size:
            self.max_box_linear_speed = max(
                self.max_box_linear_speed,
                float(np.linalg.norm(body_qd[self.float_box_body, :3])),
            )
            self.max_box_angular_speed = max(
                self.max_box_angular_speed,
                float(np.linalg.norm(body_qd[self.float_box_body, 3:])),
            )
        if body_force.size:
            self.max_box_force_norm = max(
                self.max_box_force_norm,
                float(np.linalg.norm(body_force[self.float_box_body])),
            )
        if body_force_step_avg.size:
            self.max_box_step_avg_force_norm = max(
                self.max_box_step_avg_force_norm,
                float(np.linalg.norm(body_force_step_avg[self.float_box_body])),
            )
        if sample_force.size:
            float_box_sample_force = sample_force[sample_body == self.float_box_body]
            if float_box_sample_force.size:
                self.max_sample_force_norm = max(
                    self.max_sample_force_norm,
                    float(np.linalg.norm(float_box_sample_force, axis=1).max()),
                )

    def estimate_free_surface_level(
        self,
        *,
        surface_quantile: float = 0.98,
        exclusion_scale: float = 1.5,
        wall_margin_scale: float = 2.0,
    ) -> float:
        """Estimate the free-surface height away from the floating box."""
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        particle_top = particle_q[:, 1] + particle_radius
        body_q = self.state_0.body_q.numpy()

        box_x = float(body_q[self.float_box_body, 0])
        box_z = float(body_q[self.float_box_body, 2])
        exclusion_x = float(self.box_half_extent[0]) * exclusion_scale
        exclusion_z = float(self.box_half_extent[2]) * exclusion_scale
        right_interior_x = self.current_right_wall_center_x - self.wall_thickness
        wall_margin = max(float(self.config["cell"]) * wall_margin_scale, float(self.wall_thickness))

        mask = ~((np.abs(particle_q[:, 0] - box_x) <= exclusion_x) & (np.abs(particle_q[:, 2] - box_z) <= exclusion_z))
        mask &= particle_q[:, 0] >= -self.container_half_width + wall_margin
        mask &= particle_q[:, 0] <= right_interior_x - wall_margin
        mask &= np.abs(particle_q[:, 2]) <= self.container_half_depth - wall_margin

        candidates = particle_top[mask]
        if candidates.size < max(16, self.model.particle_count // 40):
            candidates = particle_top
        return float(np.quantile(candidates, surface_quantile))

    def estimate_submerged_fraction(
        self,
        *,
        surface_quantile: float = 0.98,
        exclusion_scale: float = 1.5,
    ) -> float:
        """Estimate the current submerged volume fraction of the floating box."""
        body_q = self.state_0.body_q.numpy()
        box_center_y = float(body_q[self.float_box_body, 1])
        box_bottom = box_center_y - float(self.box_half_extent[1])
        box_height = 2.0 * float(self.box_half_extent[1])
        free_surface_y = self.estimate_free_surface_level(
            surface_quantile=surface_quantile,
            exclusion_scale=exclusion_scale,
        )
        return float(np.clip((free_surface_y - box_bottom) / max(box_height, 1.0e-8), 0.0, 1.0))

    def theoretical_submerged_fraction(self) -> float:
        """Return the buoyancy-theory submerged fraction rho_box / rho_fluid."""
        return float(
            np.clip(
                float(self.config["box_density"]) / max(float(self.config["rest_density"]), 1.0e-8),
                0.0,
                1.0,
            )
        )

    def get_buoyancy_metrics(
        self,
        *,
        surface_quantile: float = 0.98,
        exclusion_scale: float = 1.5,
    ) -> dict[str, float]:
        """Return buoyancy-related diagnostics for the floating box."""
        body_q = self.state_0.body_q.numpy()
        box_center_y = float(body_q[self.float_box_body, 1])
        box_bottom = box_center_y - float(self.box_half_extent[1])
        box_top = box_center_y + float(self.box_half_extent[1])
        free_surface_y = self.estimate_free_surface_level(
            surface_quantile=surface_quantile,
            exclusion_scale=exclusion_scale,
        )
        measured_submerged_fraction = self.estimate_submerged_fraction(
            surface_quantile=surface_quantile,
            exclusion_scale=exclusion_scale,
        )
        theoretical_submerged_fraction = self.theoretical_submerged_fraction()
        return {
            "free_surface_y": free_surface_y,
            "box_bottom_y": box_bottom,
            "box_top_y": box_top,
            "measured_submerged_fraction": measured_submerged_fraction,
            "theoretical_submerged_fraction": theoretical_submerged_fraction,
            "submerged_fraction_error": measured_submerged_fraction - theoretical_submerged_fraction,
        }

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            current_time = self.sim_time + substep * self.sim_dt
            self._set_moving_wall_state(self.state_0, current_time)
            wp.launch(
                scale_velocities,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.velocity_damping],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, control=None, contacts=self.contacts, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            reset_down = bool(self.viewer.is_key_down("r"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._update_pool_wireframe()
        if self.enable_runtime_diagnostics:
            self._record_diagnostics()
        self.sim_time += self.frame_dt

    def test_final(self):
        if not self.wall_motion_enabled:
            buoyancy = self.get_buoyancy_metrics(surface_quantile=0.95)
            free_surface_y = float(buoyancy["free_surface_y"])
            box_bottom_y = float(buoyancy["box_bottom_y"])
            measured_submerged_fraction = float(buoyancy["measured_submerged_fraction"])
            assert self.states_remain_finite, "simulation produced non-finite particle/body state"
            assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
            assert self.max_density_ratio < 2.0, (
                f"static buoyancy density ratio is too large: rho/rho0={self.max_density_ratio:.3f}"
            )
            assert self.max_particle_speed < 8.0, f"particle speed stayed too high: vmax={self.max_particle_speed:.3f}"
            assert self.max_box_linear_speed < 4.0, f"box did not settle enough: vmax={self.max_box_linear_speed:.3f}"
            assert 0.0 < measured_submerged_fraction <= 1.0, (
                "static buoyancy measurement did not produce a valid submerged fraction: "
                f"fraction={measured_submerged_fraction:.3f}"
            )
            assert box_bottom_y < free_surface_y, (
                "floating box never entered the fluid during no-wall-motion validation mode: "
                f"box_bottom={box_bottom_y:.4f}, free_surface={free_surface_y:.4f}"
            )
            return

        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        particle_top = particle_q[:, 1] + particle_radius
        body_q = self.state_0.body_q.numpy()
        box_center_y = float(body_q[self.float_box_body, 1])
        box_bottom = box_center_y - float(self.box_half_extent[1])
        box_top = box_center_y + float(self.box_half_extent[1])
        upper_bulk_level = float(np.quantile(particle_top, 0.25))
        crest_level = float(np.quantile(particle_top, 0.95))
        right_interior_x = self.current_right_wall_center_x - self.wall_thickness
        max_x = float(np.max(particle_q[:, 0] + particle_radius))
        min_x = float(np.min(particle_q[:, 0] - particle_radius))
        max_z = float(np.max(np.abs(particle_q[:, 2]) + particle_radius))
        min_y = float(np.min(particle_q[:, 1] - particle_radius))
        contact_margin = float(self.config["radius_mean"])

        expected_min_box_shift = float(self.config["expected_min_box_shift"])
        expected_min_box_force_norm = float(self.config["expected_min_box_force_norm"])
        expected_min_sample_force_norm = float(self.config["expected_min_sample_force_norm"])
        max_box_shift = max(
            abs(self.min_box_x - self.initial_box_x),
            abs(self.max_box_x - self.initial_box_x),
        )

        assert self.states_remain_finite, "simulation produced non-finite particle/body state"
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_density_ratio < 2.5, f"peak density ratio is too large: rho/rho0={self.max_density_ratio:.3f}"
        assert self.max_particle_speed < 15.0, f"particle speed blew up: vmax={self.max_particle_speed:.3f}"
        assert self.max_box_linear_speed < 8.0, f"float box linear speed blew up: vmax={self.max_box_linear_speed:.3f}"
        assert self.max_box_angular_speed < 30.0, (
            f"float box angular speed blew up: wmax={self.max_box_angular_speed:.3f}"
        )
        max_box_reaction_norm = max(self.max_box_force_norm, self.max_box_step_avg_force_norm)
        assert max_box_reaction_norm > expected_min_box_force_norm, "floating box never received body reaction"
        assert self.max_sample_force_norm > expected_min_sample_force_norm, (
            "floating-box boundary samples never received reaction forces"
        )
        assert max_box_shift > expected_min_box_shift, (
            f"floating box barely moved under fluid forcing: shift={max_box_shift:.4f}, "
            f"initial_x={self.initial_box_x:.4f}, min_x={self.min_box_x:.4f}, max_x={self.max_box_x:.4f}"
        )
        assert box_top > upper_bulk_level, (
            f"floating box fell below the upper fluid band: top={box_top:.4f}, upper_bulk={upper_bulk_level:.4f}"
        )
        assert box_bottom < crest_level + contact_margin, (
            f"floating box lost fluid contact: bottom={box_bottom:.4f}, "
            f"crest_level={crest_level:.4f}, margin={contact_margin:.4f}"
        )
        assert min_y >= -0.02, f"particles penetrated the floor too much: y={min_y:.4f}"
        assert min_x >= -self.container_half_width - 0.03, f"particles escaped along -x: {min_x:.4f}"
        assert max_x <= right_interior_x + 0.03, f"particles escaped through the moving wall: {max_x:.4f}"
        assert max_z <= self.container_half_depth + 0.03, f"particles escaped along z: {max_z:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/moving_wall_pool_wireframe",
            self.pool_wire_starts,
            self.pool_wire_ends,
            colors=self.container_wire_color,
            width=0.01,
        )
        self.viewer.log_shapes(
            "/fsi/moving_wall_float_box_body",
            newton.GeoType.BOX,
            tuple(float(self.box_half_extent[i]) for i in range(3)),
            self._body_xforms([self.float_box_body]),
            self.float_box_color,
            self.rigid_material,
        )
        self.viewer.log_shapes(
            "/fsi/moving_wall_pusher_body",
            newton.GeoType.BOX,
            (
                self.wall_thickness,
                self.wall_half_height + self.wall_thickness,
                self.container_half_depth + self.wall_thickness,
            ),
            self._body_xforms([self.moving_wall_body]),
            self.moving_wall_color,
            self.rigid_material,
        )
        self.viewer.log_points(
            "/fsi/moving_wall_ipbf_particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/moving_wall_boundary_samples",
            points=self.boundary_model.sample_x_world,
            radii=self.boundary_radii,
            colors=self.boundary_colors,
            hidden=not (self.viewer.show_particles and self.show_boundary_samples),
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
