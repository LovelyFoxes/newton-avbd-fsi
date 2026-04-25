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
# Example Fluid Block Flop
#
# Two flat square fluid slabs are stacked in a closed box. A large, shallow
# slab covers most of the floor, while a smaller slab starts above it and
# falls downward under gravity. This public comparison version keeps the
# scene parameters fixed and only swaps between IPBF and PBF.
#
# Command: python -m newton.examples fluid_block_flop
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.fluid.common import (
    add_fluid_solver_argument,
    add_shared_fluid_tuning_arguments,
    apply_shared_fluid_tuning_overrides,
    clamp_particles_to_box,
    create_fluid_solver,
    register_fluid_solver_attributes,
    shared_shape_contact_settings,
)
from newton.examples.ipbf.common import (
    build_box_wireframe,
    get_box_center,
    get_particle_grid_origin_from_center,
    scale_velocities,
)
from newton.solvers import FSIBoundaryModel


class Example:
    """Closed-box stacked slab flop test for IPBF/PBF comparison."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_fluid_solver_argument(parser)
        add_shared_fluid_tuning_arguments(parser)
        parser.add_argument(
            "--include-static-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Include sampled static walls in the fluid density and local dissipation paths.",
        )
        parser.add_argument(
            "--boundary-sample-spacing",
            type=float,
            default=None,
            help="Static boundary sample spacing [m]. Defaults to the particle spacing.",
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
            help="Boundary-sample viscosity coefficient used for local wall dissipation.",
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
            help="Boundary-sample XSPH coefficient used for local wall velocity smoothing.",
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
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.42,
                "container_half_depth": 0.42,
                "wall_half_height": 0.75,
                "bottom_dim_x": 10,
                "bottom_dim_y": 3,
                "bottom_dim_z": 10,
                "top_dim_x": 7,
                "top_dim_y": 3,
                "top_dim_z": 7,
                "cell": 0.06,
                "mass": 0.23,
                "radius_mean": 0.024,
                "smoothing_radius": 0.1,
                "bottom_center_offset_y": -0.63,
                "top_center_offset_y": 0.24,
                "iterations": 2,
                "sim_substeps": 4,
                "velocity_damping": 0.992,
                "viscosity_coefficient": 0.002,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.004,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "boundary_sample_spacing": None,
                "compliance": 1.0e-5,
                "ipbf_relaxation": 0.5,
                "pbf_relaxation": 1.0,
                "lambda_regularization": 1.0e-6,
                "use_constraint_clamp": True,
            }
            return apply_shared_fluid_tuning_overrides(self.args, config)

        config = {
            "container_half_width": 0.75,
            "container_half_depth": 0.75,
            "wall_half_height": 1.2,
            "bottom_dim_x": 96,
            "bottom_dim_y": 12,
            "bottom_dim_z": 96,
            "top_dim_x": 72,
            "top_dim_y": 8,
            "top_dim_z": 96,
            "cell": 0.014,
            "mass": 0.002744,
            "radius_mean": 0.006,
            "smoothing_radius": 0.025,
            "bottom_center_offset_y": -1.137,
            "top_center_offset_y": 0.672,
            "iterations": 2,
            "sim_substeps": 4,
            "velocity_damping": 1.0,
            "viscosity_coefficient": 0.0025,
            "viscosity_boundary_coefficient": 0.005,
            "xsph_coefficient": 0.005,
            "xsph_boundary_coefficient": 0.0,
            "boundary_velocity_damping": 1.0,
            "static_boundary_weight": 0.10,
            "include_static_boundary_samples": True,
            "boundary_sample_spacing": None,
            "compliance": 1.0e-5,
            "ipbf_relaxation": 0.5,
            "pbf_relaxation": 1.0,
            "lambda_regularization": 1.0e-6,
            "use_constraint_clamp": True,
        }
        return apply_shared_fluid_tuning_overrides(self.args, config)

    def _get_arg_or_scene_value(self, name: str, scene: dict[str, object]) -> object:
        value = getattr(self.args, name, None)
        if value is not None:
            return value
        return scene[name]

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self.fluid_solver_name = str(getattr(self.args, "fluid_solver", "ipbf"))
        self._reset_key_prev = False

        scene = self._get_scene_config()
        contact_settings = shared_shape_contact_settings(particle_radius=float(scene["radius_mean"]))
        self.sim_substeps = int(scene["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(scene["container_half_width"])
        self.container_half_depth = float(scene["container_half_depth"])
        self.wall_half_height = float(scene["wall_half_height"])
        self.wall_thickness = float(contact_settings["wall_thickness"])
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(self._get_arg_or_scene_value("velocity_damping", scene))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        register_fluid_solver_attributes(builder, self.fluid_solver_name)

        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.margin = float(contact_settings["shape_margin"])
        builder.default_shape_cfg.gap = float(contact_settings["shape_gap"])
        self._add_container(builder)

        box_center = get_box_center(wall_half_height=self.wall_half_height, floor_y=self.floor_y)

        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=(
                    box_center[0],
                    box_center[1] + float(scene["bottom_center_offset_y"]),
                    box_center[2],
                ),
                dim_x=scene["bottom_dim_x"],
                dim_y=scene["bottom_dim_y"],
                dim_z=scene["bottom_dim_z"],
                cell_x=scene["cell"],
                cell_y=scene["cell"],
                cell_z=scene["cell"],
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=scene["bottom_dim_x"],
            dim_y=scene["bottom_dim_y"],
            dim_z=scene["bottom_dim_z"],
            cell_x=scene["cell"],
            cell_y=scene["cell"],
            cell_z=scene["cell"],
            mass=scene["mass"],
            jitter=0.0,
            radius_mean=scene["radius_mean"],
        )
        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=(
                    box_center[0],
                    box_center[1] + float(scene["top_center_offset_y"]),
                    box_center[2],
                ),
                dim_x=scene["top_dim_x"],
                dim_y=scene["top_dim_y"],
                dim_z=scene["top_dim_z"],
                cell_x=scene["cell"],
                cell_y=scene["cell"],
                cell_z=scene["cell"],
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=scene["top_dim_x"],
            dim_y=scene["top_dim_y"],
            dim_z=scene["top_dim_z"],
            cell_x=scene["cell"],
            cell_y=scene["cell"],
            cell_z=scene["cell"],
            mass=scene["mass"],
            jitter=0.0,
            radius_mean=scene["radius_mean"],
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.boundary_model = None
        if bool(self._get_arg_or_scene_value("include_static_boundary_samples", scene)):
            boundary_spacing = self._get_arg_or_scene_value("boundary_sample_spacing", scene)
            if boundary_spacing is None:
                boundary_spacing = scene["cell"]
            self.boundary_model = FSIBoundaryModel(
                self.model,
                spacing=float(boundary_spacing),
                support_radius=float(scene["smoothing_radius"]),
                include_static=True,
                include_dynamic=False,
                device=self.model.device,
            )

        self.solver = create_fluid_solver(
            model=self.model,
            fluid_solver_name=self.fluid_solver_name,
            scene={
                **scene,
                "viscosity_coefficient": float(self._get_arg_or_scene_value("viscosity_coefficient", scene)),
                "viscosity_boundary_coefficient": float(
                    self._get_arg_or_scene_value("viscosity_boundary_coefficient", scene)
                ),
                "xsph_coefficient": float(self._get_arg_or_scene_value("xsph_coefficient", scene)),
                "xsph_boundary_coefficient": float(self._get_arg_or_scene_value("xsph_boundary_coefficient", scene)),
                "boundary_velocity_damping": float(self._get_arg_or_scene_value("boundary_velocity_damping", scene)),
                "static_boundary_weight": float(self._get_arg_or_scene_value("static_boundary_weight", scene)),
            },
            boundary_model=self.boundary_model,
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=float(contact_settings["soft_contact_margin"]),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.15, 0.55, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.box_wire_starts, self.box_wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=self.container_half_width,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
            include_top=True,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(2.75, 2.5, 3.25),
            pitch=-22.0,
            yaw=-132.0,
        )

        self.initial_mean_y = 0.0
        self.initial_std_x = 0.0
        self.reset()
        self.capture()

    def _add_container(self, builder: newton.ModelBuilder) -> None:
        wall_t = self.wall_thickness
        hx = self.container_half_width
        hz = self.container_half_depth
        hy = self.wall_half_height

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, -wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 2.0 * hy + wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((hx + wall_t, hy, 0.0), wp.quat_identity()),
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-(hx + wall_t), hy, 0.0), wp.quat_identity()),
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, hz + wall_t), wp.quat_identity()),
            hx=hx,
            hy=hy + wall_t,
            hz=wall_t,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, -(hz + wall_t)), wp.quat_identity()),
            hx=hx,
            hy=hy + wall_t,
            hz=wall_t,
        )

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self.contacts.clear()
        particle_q = self.state_0.particle_q.numpy()
        self.initial_mean_y = float(np.mean(particle_q[:, 1]))
        self.initial_std_x = float(np.std(particle_q[:, 0]))
        self.viewer._paused = True

    def capture(self):
        # Keep capture disabled so the IPBF and PBF variants follow the same
        # execution path inside the shared comparison examples.
        self.graph = None

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
            wp.launch(
                clamp_particles_to_box,
                dim=self.model.particle_count,
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.model.particle_radius,
                    self.container_half_width,
                    self.floor_y,
                    self.top_y,
                    self.container_half_depth,
                ],
                device=self.model.device,
            )
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
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        particle_speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)

        max_x = np.max(np.abs(particle_q[:, 0]) + particle_radius)
        max_z = np.max(np.abs(particle_q[:, 2]) + particle_radius)
        min_y = np.min(particle_q[:, 1] - particle_radius)
        max_y = np.max(particle_q[:, 1] + particle_radius)
        mean_y = float(np.mean(particle_q[:, 1]))
        std_x = float(np.std(particle_q[:, 0]))
        max_speed = np.max(particle_speed)

        assert max_x <= self.container_half_width + 0.02, f"particles escaped along x: {max_x:.3f}"
        assert max_z <= self.container_half_depth + 0.02, f"particles escaped along z: {max_z:.3f}"
        assert min_y >= -0.02, f"particles penetrated the floor: {min_y:.3f}"
        assert max_y <= self.top_y + 0.02, f"particles penetrated the ceiling: {max_y:.3f}"
        assert mean_y < self.initial_mean_y - 0.08, (
            f"upper slab did not fall enough: {mean_y:.3f} vs. {self.initial_mean_y:.3f}"
        )
        assert std_x > self.initial_std_x * 1.05, (
            f"stacked slabs did not spread laterally enough: {std_x:.3f} vs. {self.initial_std_x:.3f}"
        )
        assert max_speed <= 2.2, f"particles retained excessive speed: {max_speed:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fluid/container_wireframe",
            self.box_wire_starts,
            self.box_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_points(
            "/fluid/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
