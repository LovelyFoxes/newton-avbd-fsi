###########################################################################
# Example Basic IPBF Particles
#
# Minimal viewer example for the current IPBF solver shell.
# At this stage the solver only performs inertial prediction, initializes
# iterative buffers, builds the particle hash grid, and writes the predicted
# positions/velocities back to the output state.
#
# Command: python -m newton.examples basic_ipbf_particles
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton.solvers import SolverIPBF


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

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particle_grid(
            pos=wp.vec3(-0.225, 1.25, -0.225),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.15,
            cell_y=0.15,
            cell_z=0.15,
            mass=1.0,
            jitter=0.0,
            radius_mean=0.05,
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.15,
                iterations=2,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.15, 0.65, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(3.0, 1.6, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )

        self.reset()
        self.capture()

    def gui(self, ui):
       if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
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
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
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
        newton.examples.test_particle_state(
            self.state_0,
            "particles fall under gravity",
            lambda q, qd: q[1] < 1.25 and qd[1] < 0.0,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
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
