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
# Example IPBF Double Dam Break
#
# Two tall fluid columns collapse inward from opposite sides of a closed box.
# The scene mirrors the "double dam break" engineering test shown in the
# IPBF paper/video and keeps the public wireframe/blue-particle presentation
# used by the other Newton IPBF examples.
#
# Command: python -m newton.examples ipbf_double_dam_break
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
    get_symmetric_wall_aligned_center_offset_x,
    scale_velocities,
)
from newton.solvers import FSIBoundaryModel, SolverIPBF


class Example:
    """Closed-box double dam break container for the IPBF solver."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--include-static-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Include sampled static walls in the IPBF boundary density and local dissipation paths.",
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

    def _get_particle_block_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            return {
                "dim_x": 6,
                "dim_y": 14,
                "dim_z": 6,
                "cell": 0.07,
                "mass": 0.45,
                "radius_mean": 0.028,
                "smoothing_radius": 0.12,
                "column_center_offset_x": None,
                "column_wall_gap_x": 0.04,
                "bottom_clearance_y": 0.035,
                "iterations": 6,
                "sim_substeps": 6,
                "velocity_damping": 0.997,
                "viscosity_coefficient": 0.002,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.004,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "boundary_sample_spacing": None,
            }

        return {
            "dim_x": 28,
            "dim_y": 84,
            "dim_z": 12,
            "cell": 0.022,
            "mass": 0.010648,
            "radius_mean": 0.0095,
            "smoothing_radius": 0.04,
            "column_center_offset_x": None,
            "column_wall_gap_x": 0.03,
            "bottom_clearance_y": 0.022,
            "iterations": 4,
            "sim_substeps": 4,
            "velocity_damping": 1.0,
            "viscosity_coefficient": 0.004,
            "viscosity_boundary_coefficient": 0.000,
            "xsph_coefficient": 0.0008,
            "xsph_boundary_coefficient": 0.0,
            "boundary_velocity_damping": 0.995,
            "static_boundary_weight": 0.10,
            "include_static_boundary_samples": True,
            "boundary_sample_spacing": None,
        }

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
        self._reset_key_prev = False

        self.container_half_width = 1.08
        self.container_half_depth = 0.2
        self.wall_thickness = 0.05
        self.wall_half_height = 0.95
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        particle_block = self._get_particle_block_config()
        self.sim_substeps = int(particle_block["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.velocity_damping = float(self._get_arg_or_scene_value("velocity_damping", particle_block))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)

        _, half_span_y, _ = get_particle_grid_half_span(
            dim_x=particle_block["dim_x"],
            dim_y=particle_block["dim_y"],
            dim_z=particle_block["dim_z"],
            cell_x=particle_block["cell"],
            cell_y=particle_block["cell"],
            cell_z=particle_block["cell"],
        )
        center_offset_x = particle_block["column_center_offset_x"]
        if center_offset_x is None:
            center_offset_x = get_symmetric_wall_aligned_center_offset_x(
                container_half_width=self.container_half_width,
                dim_x=particle_block["dim_x"],
                cell_x=particle_block["cell"],
                gap_x=particle_block["column_wall_gap_x"],
            )
        center_y = self.floor_y + particle_block["bottom_clearance_y"] + half_span_y
        left_origin = get_particle_grid_origin_from_center(
            center=(-float(center_offset_x), center_y, 0.0),
            dim_x=particle_block["dim_x"],
            dim_y=particle_block["dim_y"],
            dim_z=particle_block["dim_z"],
            cell_x=particle_block["cell"],
            cell_y=particle_block["cell"],
            cell_z=particle_block["cell"],
        )
        right_origin = get_particle_grid_origin_from_center(
            center=(float(center_offset_x), center_y, 0.0),
            dim_x=particle_block["dim_x"],
            dim_y=particle_block["dim_y"],
            dim_z=particle_block["dim_z"],
            cell_x=particle_block["cell"],
            cell_y=particle_block["cell"],
            cell_z=particle_block["cell"],
        )

        for pos in (left_origin, right_origin):
            builder.add_particle_grid(
                pos=pos,
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=particle_block["dim_x"],
                dim_y=particle_block["dim_y"],
                dim_z=particle_block["dim_z"],
                cell_x=particle_block["cell"],
                cell_y=particle_block["cell"],
                cell_z=particle_block["cell"],
                mass=particle_block["mass"],
                jitter=0.0,
                radius_mean=particle_block["radius_mean"],
            )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.boundary_model = None
        if bool(self._get_arg_or_scene_value("include_static_boundary_samples", particle_block)):
            boundary_spacing = self._get_arg_or_scene_value("boundary_sample_spacing", particle_block)
            if boundary_spacing is None:
                boundary_spacing = particle_block["cell"]
            self.boundary_model = FSIBoundaryModel(
                self.model,
                spacing=float(boundary_spacing),
                support_radius=float(particle_block["smoothing_radius"]),
                include_static=True,
                include_dynamic=False,
                device=self.model.device,
            )

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=particle_block["smoothing_radius"],
                iterations=particle_block["iterations"],
                relaxation=0.5,
                viscosity_coefficient=float(self._get_arg_or_scene_value("viscosity_coefficient", particle_block)),
                viscosity_boundary_coefficient=float(
                    self._get_arg_or_scene_value("viscosity_boundary_coefficient", particle_block)
                ),
                xsph_coefficient=float(self._get_arg_or_scene_value("xsph_coefficient", particle_block)),
                xsph_boundary_coefficient=float(
                    self._get_arg_or_scene_value("xsph_boundary_coefficient", particle_block)
                ),
                boundary_velocity_damping=float(
                    self._get_arg_or_scene_value("boundary_velocity_damping", particle_block)
                ),
                fsi_static_boundary_weight=float(
                    self._get_arg_or_scene_value("static_boundary_weight", particle_block)
                ),
            ),
            boundary_model=self.boundary_model,
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=0.06,
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
            pos=wp.vec3(2.15, 2.0, 2.2),
            pitch=-22.0,
            yaw=-136.0,
        )

        self.initial_mean_abs_x = 0.0
        self.initial_center_band_count = 0
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
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 2.0 * hy + wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
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
        self.initial_mean_abs_x = float(np.mean(np.abs(particle_q[:, 0])))
        self.initial_center_band_count = int(np.count_nonzero(np.abs(particle_q[:, 0]) <= 0.15))
        self.viewer._paused = True

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
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
        mean_abs_x = float(np.mean(np.abs(particle_q[:, 0])))
        center_band_count = int(np.count_nonzero(np.abs(particle_q[:, 0]) <= 0.15))
        max_speed = np.max(particle_speed)

        assert max_x <= self.container_half_width + 0.02, f"particles escaped along x: {max_x:.3f}"
        assert max_z <= self.container_half_depth + 0.02, f"particles escaped along z: {max_z:.3f}"
        assert min_y >= -0.02, f"particles penetrated the floor: {min_y:.3f}"
        assert max_y <= self.top_y + 0.02, f"particles penetrated the ceiling: {max_y:.3f}"
        assert center_band_count >= self.initial_center_band_count + max(60, self.model.particle_count // 10), (
            "dam-break columns did not send enough particles into the center gap"
        )
        assert mean_abs_x <= self.initial_mean_abs_x + 0.28, (
            f"dam-break columns spread outward too aggressively: {mean_abs_x:.3f} vs. {self.initial_mean_abs_x:.3f}"
        )
        assert max_speed <= 3.6, f"particles retained excessive speed: {max_speed:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/ipbf/container_wireframe",
            self.box_wire_starts,
            self.box_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_points(
            "/ipbf/particles",
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
