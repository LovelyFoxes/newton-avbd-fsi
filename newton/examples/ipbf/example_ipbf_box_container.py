###########################################################################
# Example IPBF Box Container
#
# Minimal static-container example for the current IPBF solver.
# A particle block is dropped into an open box made from static collision
# shapes, while the solver projects particles back out of soft contacts.
#
# Command: python -m newton.examples ipbf_box_container
#
###########################################################################

from __future__ import annotations

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


class Example:
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

        self.container_half_width = 0.55
        self.container_half_depth = 0.55
        self.wall_thickness = 0.05
        self.wall_half_height = 0.45
        self.initial_particle_velocity = wp.vec3(0.9, 0.0, 0.35)
        # The current IPBF boundary handling uses post-update positional projection
        # without dedicated wall friction, so a small global velocity damping keeps
        # the demo bounded and lets the particle block settle inside the box.
        self.velocity_damping = 0.989

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)

        builder.add_particle_grid(
            pos=wp.vec3(-0.24, 0.12, -0.24),
            rot=wp.quat_identity(),
            vel=self.initial_particle_velocity,
            dim_x=7,
            dim_y=8,
            dim_z=7,
            cell_x=0.075,
            cell_y=0.075,
            cell_z=0.075,
            mass=0.5,
            jitter=0.0,
            radius_mean=0.03,
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.12,
                iterations=4,
                relaxation=0.5,
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

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(1.6, 1.85, 1.6),
            pitch=-40.0,
            yaw=-135.0,
        )

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

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self.contacts.clear()
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
        max_speed = np.max(particle_speed)

        assert max_x <= self.container_half_width + 0.02, f"particles escaped along x: {max_x:.3f}"
        assert max_z <= self.container_half_depth + 0.02, f"particles escaped along z: {max_z:.3f}"
        assert min_y >= -0.02, f"particles penetrated the floor: {min_y:.3f}"
        assert max_speed <= 2.0, f"particles retained excessive speed: {max_speed:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
