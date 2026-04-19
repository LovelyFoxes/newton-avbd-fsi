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
# Example FSI IPBF VBD Three Sphere Buoys
#
# Single-fluid AVBD/IPBF tank scene with three dynamic sphere buoys of
# different densities. The scene validates that analytic sphere boundary
# samples participate in the same IPBF density / pressure-reaction / AVBD
# rigid-body feedback path as the earlier box FSI examples.
#
# Command: python -m newton.examples fsi_ipbf_vbd_three_sphere_buoys
#
###########################################################################

from __future__ import annotations

import argparse

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
    """Open-top pool scene with three sphere buoys of different densities."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used by the three-buoy scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of IPBF/AVBD iteration pairs used per simulation substep.",
        )
        parser.add_argument("--ipbf-iterations", type=int, default=None)
        parser.add_argument("--sim-substeps", type=int, default=None)
        parser.add_argument("--sphere-radius", type=float, default=None, help="Override all buoy radii [m].")
        parser.add_argument(
            "--sphere-bottom-gap",
            type=float,
            default=None,
            help="Override the initial gap between the water surface and each sphere bottom [m].",
        )
        parser.add_argument(
            "--sphere-densities",
            type=float,
            nargs=3,
            default=None,
            metavar=("LIGHT", "MID", "HEAVY"),
            help="Override the three sphere densities [kg/m^3].",
        )
        parser.add_argument("--projection-reaction-relaxation", type=float, default=None)
        parser.add_argument("--velocity-reaction-relaxation", type=float, default=None)
        parser.add_argument("--pressure-reaction-relaxation", type=float, default=None)
        parser.add_argument(
            "--water-render-radius-scale",
            type=float,
            default=None,
            help="Visual-only radius multiplier for rendered water particles.",
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
            help="Diagnostic weight applied to static boundary samples.",
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
                "dynamic-shape-volume",
                "dynamic-shape-surface-quadrature",
                "dynamic-shape-surface-thickness",
            ],
            default="dynamic-shape-surface-quadrature",
            help="Hydrostatic boundary-volume model used for dynamic sphere boundary samples.",
        )
        parser.add_argument("--enable-runtime-diagnostics", action="store_true")
        return parser

    @staticmethod
    def _parse_hydrostatic_volume_mode(mode: str) -> FSIBoundaryModel.HydrostaticVolumeMode:
        mode_map = {
            "raw": FSIBoundaryModel.HydrostaticVolumeMode.NONE,
            "dynamic-shape-volume": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_VOLUME,
            "dynamic-shape-surface-quadrature": (
                FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE
            ),
            "dynamic-shape-surface-thickness": (FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_THICKNESS),
        }
        return mode_map[str(mode)]

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.36,
                "container_half_depth": 0.28,
                "wall_half_height": 0.72,
                "pool_dim_x": 16,
                "pool_dim_y": 12,
                "pool_dim_z": 12,
                "cell": 0.04,
                "mass": 0.064,
                "radius_mean": 0.016,
                "smoothing_radius": 0.075,
                "pool_bottom_clearance": 0.03,
                "sphere_radius": 0.055,
                "sphere_bottom_gap": 0.025,
                "sphere_densities": (300.0, 750.0, 1250.0),
                "rest_density": 1000.0,
                "ipbf_iterations": 6,
                "sim_substeps": 6,
                "velocity_damping": 0.996,
                "viscosity_coefficient": 0.002,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.004,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "rigid_iterations": 2,
                "boundary_spacing": 0.04,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "water_render_radius_scale": 0.45,
                "boundary_render_radius_scale": 0.30,
                "expected_min_drop": 0.02,
                "expected_min_body_force_norm": 0.05,
                "expected_min_sample_force_norm": 0.005,
                "expected_min_density_ordering_gap": 0.001,
            }
        else:
            config = {
                "container_half_width": 0.52,
                "container_half_depth": 0.39,
                "wall_half_height": 0.92,
                "pool_dim_x": 64,
                "pool_dim_y": 30,
                "pool_dim_z": 48,
                "cell": 0.014,
                "mass": 0.002744,
                "radius_mean": 0.006,
                "smoothing_radius": 0.025,
                "pool_bottom_clearance": 0.03,
                "sphere_radius": 0.07,
                "sphere_bottom_gap": 0.035,
                "sphere_densities": (250.0, 700.0, 1250.0),
                "rest_density": 1000.0,
                "ipbf_iterations": 4,
                "sim_substeps": 4,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.010,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 0.97,
                "rigid_iterations": 4,
                "boundary_spacing": 0.014,
                "static_boundary_weight": 0.10,
                "include_static_boundary_samples": True,
                "water_render_radius_scale": 0.45,
                "boundary_render_radius_scale": 0.30,
                "expected_min_drop": 0.0,
                "expected_min_body_force_norm": 0.0,
                "expected_min_sample_force_norm": 0.0,
                "expected_min_density_ordering_gap": 0.0,
            }

        overrides = {
            "ipbf_iterations": getattr(self.args, "ipbf_iterations", None),
            "sim_substeps": getattr(self.args, "sim_substeps", None),
            "sphere_radius": getattr(self.args, "sphere_radius", None),
            "sphere_bottom_gap": getattr(self.args, "sphere_bottom_gap", None),
            "water_render_radius_scale": getattr(self.args, "water_render_radius_scale", None),
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = int(value) if key.endswith("iterations") or key == "sim_substeps" else float(value)
        if getattr(self.args, "sphere_densities", None) is not None:
            config["sphere_densities"] = tuple(float(v) for v in self.args.sphere_densities)

        include_static_override = getattr(self.args, "include_static_boundary_samples", None)
        if include_static_override is not None:
            config["include_static_boundary_samples"] = bool(include_static_override)

        optional_float_overrides = {
            "velocity_damping": getattr(self.args, "velocity_damping", None),
            "viscosity_coefficient": getattr(self.args, "viscosity_coefficient", None),
            "viscosity_boundary_coefficient": getattr(self.args, "viscosity_boundary_coefficient", None),
            "xsph_coefficient": getattr(self.args, "xsph_coefficient", None),
            "xsph_boundary_coefficient": getattr(self.args, "xsph_boundary_coefficient", None),
            "boundary_velocity_damping": getattr(self.args, "boundary_velocity_damping", None),
            "static_boundary_weight": getattr(self.args, "static_boundary_weight", None),
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
        self.hydrostatic_volume_mode = self._parse_hydrostatic_volume_mode(
            getattr(self.args, "hydrostatic_volume_mode", "dynamic-shape-surface-quadrature")
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
        self.sphere_radius = float(self.config["sphere_radius"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)
        self.buoy_bodies = self._add_sphere_buoys(builder)
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
                compliance=1.0e-5,
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=0.5,
                viscosity_coefficient=float(self.config["viscosity_coefficient"]),
                viscosity_boundary_coefficient=float(self.config["viscosity_boundary_coefficient"]),
                xsph_coefficient=float(self.config["xsph_coefficient"]),
                xsph_boundary_coefficient=float(self.config["xsph_boundary_coefficient"]),
                boundary_velocity_damping=float(self.config["boundary_velocity_damping"]),
                fsi_projection_reaction_relaxation=self._reaction_override("projection", 0.0),
                fsi_velocity_projection_reaction_relaxation=self._reaction_override("velocity", 0.0),
                fsi_pressure_reaction_relaxation=self._reaction_override("pressure", 2.0),
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
            value=wp.vec3(0.48, 0.82, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.water_radii = wp.full(
            self.model.particle_count,
            value=float(self.config["radius_mean"]) * float(self.config["water_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.boundary_colors = self._build_boundary_colors()
        self.buoy_colors = wp.array(
            [
                wp.vec3(1.0, 0.78, 0.24),
                wp.vec3(0.34, 0.86, 0.34),
                wp.vec3(0.20, 0.92, 0.92),
            ],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.rigid_material = wp.array([wp.vec4(0.42, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * float(self.config["boundary_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.pool_wire_starts, self.pool_wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=self.container_half_width,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
            include_top=False,
        )

        self.initial_body_y = np.zeros(len(self.buoy_bodies), dtype=np.float32)
        self.min_body_y = np.zeros(len(self.buoy_bodies), dtype=np.float32)
        self.final_body_y = np.zeros(len(self.buoy_bodies), dtype=np.float32)
        self.initial_particle_max_y = 0.0
        self.max_particle_y = 0.0
        self.max_body_force_norm = 0.0
        self.max_body_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_body_linear_speed = 0.0
        self.max_body_angular_speed = 0.0
        self.states_remain_finite = True
        self.enable_runtime_diagnostics = bool(getattr(self.args, "enable_runtime_diagnostics", False)) or bool(
            getattr(self.args, "test", False)
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.75, 1.28, 1.95), pitch=-22.0, yaw=-132.0)
        self.reset()

    def _reaction_override(self, name: str, default: float) -> float:
        value = getattr(self.args, f"{name}_reaction_relaxation", None)
        return default if value is None else float(value)

    def _build_boundary_colors(self) -> wp.array(dtype=wp.vec3):
        sample_body = self.boundary_model.sample_body.numpy()
        colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        body_colors = (
            np.array([1.0, 0.78, 0.24], dtype=np.float32),
            np.array([0.34, 0.86, 0.34], dtype=np.float32),
            np.array([0.20, 0.92, 0.92], dtype=np.float32),
        )
        for body, color in zip(self.buoy_bodies, body_colors, strict=True):
            colors[sample_body == body] = color
        colors[sample_body < 0] = np.array([0.92, 0.94, 0.98], dtype=np.float32)
        return wp.array(colors, dtype=wp.vec3, device=self.model.device)

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

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
            xform=wp.transform((hx + wall_t, hy, 0.0), wp.quat_identity()),
            hx=wall_t,
            hy=hy + wall_t,
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

    def _add_sphere_buoys(self, builder: newton.ModelBuilder) -> list[int]:
        water_surface_y = self._compute_pool_surface_y()
        sphere_center_y = water_surface_y + float(self.config["sphere_bottom_gap"]) + self.sphere_radius
        x_positions = (-0.22, 0.0, 0.22) if bool(getattr(self.args, "test", False)) else (-0.30, 0.0, 0.30)
        densities = tuple(float(value) for value in self.config["sphere_densities"])

        bodies: list[int] = []
        for index, (x_position, density) in enumerate(zip(x_positions, densities, strict=True)):
            body = builder.add_body(
                xform=wp.transform((x_position, sphere_center_y, 0.0), wp.quat_identity()),
                label=f"fsi_sphere_buoy_{index}",
            )
            builder.add_shape_sphere(
                body=body,
                radius=self.sphere_radius,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.0),
            )
            bodies.append(body)
        return bodies

    def _add_pool(self, builder: newton.ModelBuilder) -> None:
        _, pool_half_span_y, _ = get_particle_grid_half_span(
            dim_x=int(self.config["pool_dim_x"]),
            dim_y=int(self.config["pool_dim_y"]),
            dim_z=int(self.config["pool_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
        )
        pool_center = (
            0.0,
            self.floor_y + float(self.config["pool_bottom_clearance"]) + pool_half_span_y,
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

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Static boundary samples: {'on' if self.include_static_boundary_samples else 'off'}")
        ui.text(f"Render boundary samples: {'on' if self.show_boundary_samples else 'off'}")
        ui.text(f"Hydrostatic mode: {self.args.hydrostatic_volume_mode!s}")
        ui.text(f"Sphere radius: {self.sphere_radius:.3f} m")
        ui.text(f"Sphere densities: {tuple(float(v) for v in self.config['sphere_densities'])}")
        ui.text(f"Water render radius scale: {float(self.config['water_render_radius_scale']):.2f}")

    def reset(self):
        self.sim_time = 0.0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()

        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        body_y = body_q[self.buoy_bodies, 1].astype(np.float32)
        self.initial_body_y = body_y.copy()
        self.min_body_y = body_y.copy()
        self.final_body_y = body_y.copy()
        self.initial_particle_max_y = float(np.max(particle_q[:, 1] + particle_radius))
        self.max_particle_y = self.initial_particle_max_y
        self.max_body_force_norm = 0.0
        self.max_body_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_body_linear_speed = 0.0
        self.max_body_angular_speed = 0.0
        self.states_remain_finite = True
        self.viewer._paused = True

    def _record_diagnostics(self) -> None:
        body_force = self.boundary_model.body_force.numpy()
        body_force_step_avg = self.boundary_model.body_force_step_avg.numpy()
        sample_force = self.boundary_model.sample_force.numpy()
        sample_body = self.boundary_model.sample_body.numpy()
        density = self.state_0.ipbf.density.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        particle_radius = self.model.particle_radius.numpy()

        body_y = body_q[self.buoy_bodies, 1].astype(np.float32)
        self.min_body_y = np.minimum(self.min_body_y, body_y)
        self.final_body_y = body_y.copy()
        self.max_particle_y = max(self.max_particle_y, float(np.max(particle_q[:, 1] + particle_radius)))
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(body_force).all()
            and np.isfinite(body_force_step_avg).all()
            and np.isfinite(sample_force).all()
            and np.isfinite(density).all()
            and np.isfinite(body_q).all()
            and np.isfinite(body_qd).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
        )

        if body_force.size:
            self.max_body_force_norm = max(
                self.max_body_force_norm,
                float(np.linalg.norm(body_force[self.buoy_bodies], axis=1).max()),
            )
        if body_force_step_avg.size:
            self.max_body_step_avg_force_norm = max(
                self.max_body_step_avg_force_norm,
                float(np.linalg.norm(body_force_step_avg[self.buoy_bodies], axis=1).max()),
            )
        if sample_force.size:
            buoy_mask = np.isin(sample_body, np.asarray(self.buoy_bodies, dtype=np.int32))
            buoy_sample_force = sample_force[buoy_mask]
            if buoy_sample_force.size:
                self.max_sample_force_norm = max(
                    self.max_sample_force_norm,
                    float(np.linalg.norm(buoy_sample_force, axis=1).max()),
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
            buoy_qd = body_qd[self.buoy_bodies]
            self.max_body_linear_speed = max(
                self.max_body_linear_speed,
                float(np.linalg.norm(buoy_qd[:, :3], axis=1).max()),
            )
            self.max_body_angular_speed = max(
                self.max_body_angular_speed,
                float(np.linalg.norm(buoy_qd[:, 3:], axis=1).max()),
            )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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

        self.simulate()
        if self.enable_runtime_diagnostics:
            self._record_diagnostics()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        body_y = self.state_0.body_q.numpy()[self.buoy_bodies, 1]
        max_x = float(np.max(np.abs(particle_q[:, 0]) + particle_radius))
        max_z = float(np.max(np.abs(particle_q[:, 2]) + particle_radius))
        min_particle_y = float(np.min(particle_q[:, 1] - particle_radius))
        max_reaction_norm = max(self.max_body_force_norm, self.max_body_step_avg_force_norm)

        assert self.states_remain_finite, "simulation produced non-finite particle/body state"
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_density_ratio < 2.8, f"peak density ratio is too large: rho/rho0={self.max_density_ratio:.3f}"
        assert self.max_particle_speed < 25.0, f"particle speed blew up: vmax={self.max_particle_speed:.3f}"
        assert self.max_body_linear_speed < 20.0, f"buoy linear speed blew up: vmax={self.max_body_linear_speed:.3f}"
        assert self.max_body_angular_speed < 120.0, (
            f"buoy angular speed blew up: wmax={self.max_body_angular_speed:.3f}"
        )
        assert max_reaction_norm > float(self.config["expected_min_body_force_norm"]), (
            "sphere buoys never received FSI body reaction"
        )
        assert self.max_sample_force_norm > float(self.config["expected_min_sample_force_norm"]), (
            "sphere boundary samples never received reaction forces"
        )
        assert np.any(self.min_body_y < self.initial_body_y - float(self.config["expected_min_drop"])), (
            "none of the sphere buoys entered the pool enough to exercise FSI contact"
        )
        assert body_y[0] > body_y[2] + float(self.config["expected_min_density_ordering_gap"]), (
            "light sphere did not remain above the heavy sphere in the single-fluid buoy test: "
            f"light_y={body_y[0]:.4f}, heavy_y={body_y[2]:.4f}"
        )
        assert self.max_particle_y > self.initial_particle_max_y - 2.0 * float(self.config["radius_mean"]), (
            "deep pool free-surface band collapsed too much during buoy interaction"
        )
        assert min_particle_y >= -0.02, f"particles penetrated the floor too much: y={min_particle_y:.4f}"
        assert max_x <= self.container_half_width + 0.05, f"particles escaped along x: {max_x:.4f}"
        assert max_z <= self.container_half_depth + 0.05, f"particles escaped along z: {max_z:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/three_sphere_buoys_pool_wireframe",
            self.pool_wire_starts,
            self.pool_wire_ends,
            colors=(0.88, 0.90, 0.95),
            width=0.01,
        )
        self.viewer.log_shapes(
            "/fsi/three_sphere_buoys_rigid_bodies",
            newton.GeoType.SPHERE,
            self.sphere_radius,
            self._body_xforms(self.buoy_bodies),
            self.buoy_colors,
            self.rigid_material,
        )
        self.viewer.log_points(
            "/fsi/three_sphere_buoys_ipbf_particles",
            points=self.state_0.particle_q,
            radii=self.water_radii,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/three_sphere_buoys_boundary_samples",
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
