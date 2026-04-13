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
# Example FSI IPBF VBD Loose
#
# A loose-coupled AVBD/IPBF fluid-solid interaction demo. An IPBF particle
# block moves into a dynamic AVBD box; contact projection produces an FSI
# reaction wrench that pushes the box.
#
# Command: python -m newton.examples fsi_ipbf_vbd_loose
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ipbf.common import build_box_wireframe, get_particle_grid_origin_from_center, scale_velocities
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD


class Example:
    def _get_scene_config(self) -> dict[str, float | int | wp.vec3]:
        if bool(getattr(self.args, "test", False)):
            return {
                "dim_x": 5,
                "dim_y": 4,
                "dim_z": 5,
                "cell": 0.03,
                "mass": 0.024,
                "radius": 0.013,
                "smoothing_radius": 0.052,
                "rest_density": 1000.0,
                "ipbf_iterations": 4,
                "particle_velocity": wp.vec3(1.2, 0.0, 0.0),
                "particle_center": wp.vec3(-0.22, 0.20, 0.0),
                "box_center": wp.vec3(0.17, 0.20, 0.0),
                "box_mass": 1.0,
                "boundary_spacing": 0.045,
                "fsi_reaction_relaxation": 0.004,
            }

        return {
            "dim_x": 12,
            "dim_y": 8,
            "dim_z": 10,
            "cell": 0.028,
            "mass": 0.022,
            "radius": 0.012,
            "smoothing_radius": 0.048,
            "rest_density": 1000.0,
            "ipbf_iterations": 5,
            "particle_velocity": wp.vec3(0.85, 0.0, 0.0),
            "particle_center": wp.vec3(-0.24, 0.22, 0.0),
            "box_center": wp.vec3(0.18, 0.22, 0.0),
            "box_mass": 1.5,
            "boundary_spacing": 0.04,
            "fsi_reaction_relaxation": 0.001,
        }

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False

        self.config = self._get_scene_config()
        self.channel_half_width = 0.55
        self.channel_half_depth = 0.26
        self.wall_thickness = 0.04
        self.wall_half_height = 0.22
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.box_half_extent = wp.vec3(0.045, self.wall_half_height, self.channel_half_depth)
        self.velocity_damping = 0.992

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_channel_walls(builder)
        self.box_body = builder.add_body(
            xform=wp.transform(self.config["box_center"], wp.quat_identity()),
            mass=self.config["box_mass"],
            lock_inertia=True,
            label="dynamic_fsi_box",
        )
        builder.add_shape_box(
            body=self.box_body,
            hx=float(self.box_half_extent[0]),
            hy=float(self.box_half_extent[1]),
            hz=float(self.box_half_extent[2]),
        )

        dim_x = int(self.config["dim_x"])
        dim_y = int(self.config["dim_y"])
        dim_z = int(self.config["dim_z"])
        cell = float(self.config["cell"])
        particle_center = self.config["particle_center"]
        particle_origin = get_particle_grid_origin_from_center(
            center=(float(particle_center[0]), float(particle_center[1]), float(particle_center[2])),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
        )
        builder.add_particle_grid(
            pos=particle_origin,
            rot=wp.quat_identity(),
            vel=self.config["particle_velocity"],
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            mass=float(self.config["mass"]),
            jitter=0.0,
            radius_mean=float(self.config["radius"]),
        )

        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            include_static=False,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=float(self.config["rest_density"]),
                smoothing_radius=float(self.config["smoothing_radius"]),
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=0.5,
                fsi_reaction_relaxation=float(self.config["fsi_reaction_relaxation"]),
            ),
            boundary_model=self.boundary_model,
        )
        self.solid_solver = SolverVBD(
            self.model,
            iterations=1,
            integrate_particles=False,
            fsi_boundary_model=self.boundary_model,
        )
        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=float(self.config["radius"]) * 2.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)
        self.initial_box_x = float(self.state_0.body_q.numpy()[self.box_body, 0])

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.12, 0.58, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.boundary_colors = wp.full(
            self.boundary_model.sample_count,
            value=wp.vec3(1.0, 0.68, 0.24),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * 0.18,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.channel_wire_starts, self.channel_wire_ends = build_box_wireframe(
            min_x=-self.channel_half_width,
            max_x=self.channel_half_width,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.channel_half_depth,
            max_z=self.channel_half_depth,
            device=self.model.device,
        )
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(1.1, 0.72, 1.0),
            pitch=-20.0,
            yaw=-132.0,
        )

        self.reset()
        self.capture()

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def _add_channel_walls(self, builder: newton.ModelBuilder) -> None:
        wall_t = self.wall_thickness
        hx = self.channel_half_width
        hz = self.channel_half_depth
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

    def reset(self):
        self.sim_time = 0.0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()
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
        body_q = self.state_0.body_q.numpy()
        density = self.state_0.ipbf.density.numpy()
        box_x = float(body_q[self.box_body, 0])
        max_density = float(np.max(density))

        assert box_x > self.initial_box_x + 1.0e-4, f"dynamic box was not pushed forward: x={box_x:.5f}"
        assert max_density > 0.0, "IPBF density diagnostics were not updated"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/channel_wireframe",
            self.channel_wire_starts,
            self.channel_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_points(
            "/fsi/ipbf_particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/boundary_samples",
            points=self.boundary_model.sample_x_world,
            radii=self.boundary_radii,
            colors=self.boundary_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
