# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IPBF Block Flop
#
# Two flat square fluid slabs are stacked in a closed box. A large, shallow
# slab covers most of the floor, while a smaller slab starts above it and
# falls downward under gravity. This matches the "block flop" style scene
# requested for the IPBF engineering comparisons.
#
# Command: python -m newton.examples ipbf_block_flop
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ipbf.common import build_box_wireframe, scale_velocities
from newton.solvers import SolverIPBF


class Example:
    """Closed-box stacked slab flop test for the IPBF solver."""

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            return {
                "container_half_width": 0.42,
                "container_half_depth": 0.42,
                "wall_half_height": 0.75,
                "bottom_pos": wp.vec3(-0.27, 0.03, -0.27),
                "bottom_dim_x": 10,
                "bottom_dim_y": 3,
                "bottom_dim_z": 10,
                "top_pos": wp.vec3(-0.18, 0.9, -0.18),
                "top_dim_x": 7,
                "top_dim_y": 3,
                "top_dim_z": 7,
                "cell": 0.06,
                "mass": 0.23,
                "radius_mean": 0.024,
                "smoothing_radius": 0.1,
                "iterations": 6,
                "sim_substeps": 6,
                "velocity_damping": 0.992,
                "viscosity": 0.002,
                "xsph": 0.004,
                "compliance": 1.0e-5,
            }

        return {
            "container_half_width": 1.0,
            "container_half_depth": 1.0,
            "wall_half_height": 1.2,
            "bottom_pos": wp.vec3(-0.889, 0.014, -0.889),
            "bottom_dim_x": 128,
            "bottom_dim_y": 8,
            "bottom_dim_z": 128,
            "top_pos": wp.vec3(-0.609, 1.48, -0.609),
            "top_dim_x": 88,
            "top_dim_y": 9,
            "top_dim_z": 88,
            "cell": 0.014,
            "mass": 0.002744,
            "radius_mean": 0.006,
            "smoothing_radius": 0.025,
            "iterations": 2,
            "sim_substeps": 4,
            "velocity_damping": 0.999,
            "viscosity": 0.0025,
            "xsph": 0.005,
            "compliance": 1.0e-5,
        }

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False

        scene = self._get_scene_config()
        self.sim_substeps = int(scene["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(scene["container_half_width"])
        self.container_half_depth = float(scene["container_half_depth"])
        self.wall_half_height = float(scene["wall_half_height"])
        self.wall_thickness = 0.05
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(scene["velocity_damping"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)

        builder.add_particle_grid(
            pos=scene["bottom_pos"],
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
            pos=scene["top_pos"],
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

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=scene["smoothing_radius"],
                compliance=scene["compliance"],
                iterations=scene["iterations"],
                relaxation=0.5,
                viscosity_coefficient=scene["viscosity"],
                xsph_coefficient=scene["xsph"],
            ),
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=0.05,
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
            pos=wp.vec3(2.25, 1.8, 2.7),
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
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
