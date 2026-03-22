# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IPBF Box Moving Wall
#
# Closed-box IPBF example with a kinematic right wall that periodically
# moves inward and outward to stir the particle block. The walls are not
# rendered as solid geometry; instead the example draws a wireframe box and
# a blue particle cloud so the fluid motion stays easy to inspect.
#
# Command: python -m newton.examples ipbf_box_moving_wall
#
###########################################################################

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverIPBF


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
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    starts = wp.array(corners[[i for i, _ in edges]], dtype=wp.vec3, device=device)
    ends = wp.array(corners[[j for _, j in edges]], dtype=wp.vec3, device=device)
    return starts, ends


class Example:
    """Closed IPBF box with a kinematic moving side wall."""

    def _get_particle_block_config(self) -> dict[str, float | int | wp.vec3]:
        if bool(getattr(self.args, "test", False)):
            return {
                "pos": wp.vec3(-0.78, 0.12, -0.19),
                "dim_x": 6,
                "dim_y": 7,
                "dim_z": 6,
                "cell_x": 0.075,
                "cell_y": 0.075,
                "cell_z": 0.075,
                "mass": 0.5,
                "radius_mean": 0.03,
                "smoothing_radius": 0.12,
                "iterations": 5,
            }

        return {
            "pos": wp.vec3(-0.79, 0.035, -0.53),
            "dim_x": 26,
            "dim_y": 30,
            "dim_z": 39,
            "cell_x": 0.028,
            "cell_y": 0.028,
            "cell_z": 0.028,
            "mass": 0.022,
            "radius_mean": 0.012,
            "smoothing_radius": 0.048,
            "iterations": 6,
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
        self.graph = None

        self.container_half_width = 0.9
        self.container_half_depth = 0.55
        self.wall_thickness = 0.05
        self.wall_half_height = 0.45
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = 0.994
        self.wall_travel = float(getattr(args, "wall_travel", 0.25))
        self.wall_frequency = float(getattr(args, "wall_frequency", 1.00))
        self.current_right_wall_center_x = self.container_half_width + self.wall_thickness
        particle_block = self._get_particle_block_config()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)

        builder.add_particle_grid(
            pos=particle_block["pos"],
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=particle_block["dim_x"],
            dim_y=particle_block["dim_y"],
            dim_z=particle_block["dim_z"],
            cell_x=particle_block["cell_x"],
            cell_y=particle_block["cell_y"],
            cell_z=particle_block["cell_z"],
            mass=particle_block["mass"],
            jitter=0.0,
            radius_mean=particle_block["radius_mean"],
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=particle_block["smoothing_radius"],
                iterations=particle_block["iterations"],
                relaxation=0.5,
                viscosity_coefficient=0.005,
                xsph_coefficient=0.02,
            ),
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

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(2.7, 1.75, 2.15),
            pitch=-28.0,
            yaw=-132.0,
        )

        self.reset()

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

        self.moving_wall_body = builder.add_body(
            xform=wp.transform((hx + wall_t, hy, 0.0), wp.quat_identity()),
            is_kinematic=True,
            label="moving_wall",
        )
        builder.add_shape_box(
            body=self.moving_wall_body,
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
        )

    def _moving_wall_state(self, time_s: float) -> tuple[float, float]:
        omega = 2.0 * math.pi * self.wall_frequency
        center_x = self.container_half_width + self.wall_thickness - 0.5 * self.wall_travel * (1.0 - math.cos(omega * time_s))
        velocity_x = -0.5 * self.wall_travel * omega * math.sin(omega * time_s)
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

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Wall travel: {self.wall_travel:.3f} m")
        ui.text(f"Wall frequency: {self.wall_frequency:.2f} Hz")

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self._set_moving_wall_state(self.state_0, 0.0)
        self._set_moving_wall_state(self.state_1, 0.0)
        self.contacts.clear()
        self.viewer._paused = True

    def capture(self):
        # Disable CUDA graph capture because the kinematic wall state is updated
        # from Python every substep.
        self.graph = None

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

        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        particle_speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)

        min_x = np.min(particle_q[:, 0] - particle_radius)
        max_x = np.max(particle_q[:, 0] + particle_radius)
        max_z = np.max(np.abs(particle_q[:, 2]) + particle_radius)
        min_y = np.min(particle_q[:, 1] - particle_radius)
        max_y = np.max(particle_q[:, 1] + particle_radius)
        max_speed = np.max(particle_speed)
        right_interior_x = self.current_right_wall_center_x - self.wall_thickness

        assert min_x >= -self.container_half_width - 0.02, f"particles escaped through the left wall: {min_x:.3f}"
        assert max_x <= right_interior_x + 0.02, f"particles escaped through the moving wall: {max_x:.3f}"
        assert max_z <= self.container_half_depth + 0.02, f"particles escaped along z: {max_z:.3f}"
        assert min_y >= -0.02, f"particles penetrated the floor: {min_y:.3f}"
        assert max_y <= self.top_y + 0.02, f"particles penetrated the ceiling: {max_y:.3f}"
        assert max_speed <= 2.8, f"particles retained excessive speed: {max_speed:.3f}"

    def render(self):
        right_interior_x = self.current_right_wall_center_x - self.wall_thickness
        wire_starts, wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=right_interior_x,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/ipbf/container_wireframe",
            wire_starts,
            wire_ends,
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
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--wall-travel",
        type=float,
        default=0.25,
        help="Peak inward travel of the moving wall [m].",
    )
    parser.add_argument(
        "--wall-frequency",
        type=float,
        default=1.00,
        help="Oscillation frequency of the moving wall [Hz].",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
