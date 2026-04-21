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
# Example FSI IPBF VBD Interlinked
#
# An interlinked AVBD/IPBF fluid-solid interaction demo. An IPBF particle
# block starts close to a dynamic AVBD piston. Boundary density and
# pressure-gradient reaction, rather than initial penetration, push the piston.
#
# Command: python -m newton.examples fsi_ipbf_vbd_interlinked
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
    """Interlinked AVBD/IPBF pressure-coupling demo."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used for the pressure-coupling demo.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=2,
            help="Number of outer IPBF/AVBD feedback passes for interlinked mode.",
        )
        parser.add_argument(
            "--pressure-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to pressure-gradient reaction forces.",
        )
        parser.add_argument(
            "--projection-reaction-relaxation",
            type=float,
            default=0.0,
            help="Scale applied to shape-projection reaction forces. Defaults to zero to isolate pressure coupling.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, float | int | wp.vec3]:
        pressure_relaxation = getattr(self.args, "pressure_reaction_relaxation", None)

        if bool(getattr(self.args, "test", False)):
            return {
                "dim_x": 3,
                "dim_y": 3,
                "dim_z": 3,
                "cell": 0.028,
                "mass": 0.015,
                "radius": 0.011,
                "smoothing_radius": 0.12,
                "rest_density": 1000.0,
                "ipbf_iterations": 2,
                "ipbf_relaxation": 0.05,
                "viscosity": 0.0,
                "xsph": 0.0,
                "particle_center": wp.vec3(0.018, 0.16, 0.0),
                "particle_velocity": wp.vec3(0.0, 0.0, 0.0),
                "piston_center": wp.vec3(0.075, 0.16, 0.0),
                "piston_mass": 0.9,
                "rigid_iterations": 2,
                "boundary_spacing": 0.05,
                "pressure_reaction_relaxation": 0.08 if pressure_relaxation is None else float(pressure_relaxation),
            }

        return {
            "dim_x": 5,
            "dim_y": 5,
            "dim_z": 5,
            "cell": 0.026,
            "mass": 0.011,
            "radius": 0.010,
            "smoothing_radius": 0.12,
            "rest_density": 1000.0,
            "ipbf_iterations": 4,
            "ipbf_relaxation": 0.001,
            "viscosity": 0.0005,
            "xsph": 0.004,
            "particle_center": wp.vec3(-0.030, 0.18, 0.0),
            "particle_velocity": wp.vec3(0.04, 0.0, 0.0),
            "piston_center": wp.vec3(0.075, 0.18, 0.0),
            "piston_mass": 0.05,
            "rigid_iterations": 4,
            "boundary_spacing": 0.05,
            "pressure_reaction_relaxation": 5.0 if pressure_relaxation is None else float(pressure_relaxation),
        }

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False

        self.config = self._get_scene_config()
        self.domain_min = wp.vec3(-0.22, 0.03, -0.16)
        self.domain_max = wp.vec3(0.18, 0.32, 0.16)
        self.wall_thickness = 0.03
        self.piston_half_extent = wp.vec3(0.025, 0.12, 0.12)
        self.velocity_damping = 0.994

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0

        self._add_domain_walls(builder)
        self.piston_body = builder.add_body(
            xform=wp.transform(self.config["piston_center"], wp.quat_identity()),
            mass=float(self.config["piston_mass"]),
            lock_inertia=True,
            label="pressure_coupled_piston",
        )
        builder.add_shape_box(
            body=self.piston_body,
            hx=float(self.piston_half_extent[0]),
            hy=float(self.piston_half_extent[1]),
            hz=float(self.piston_half_extent[2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_particle_collision=False,
                has_shape_collision=False,
            ),
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
                relaxation=float(self.config["ipbf_relaxation"]),
                viscosity_coefficient=float(self.config["viscosity"]),
                xsph_coefficient=float(self.config["xsph"]),
                fsi_projection_reaction_relaxation=float(getattr(self.args, "projection_reaction_relaxation", 0.0)),
                fsi_velocity_projection_reaction_relaxation=float(
                    getattr(self.args, "projection_reaction_relaxation", 0.0)
                ),
                fsi_pressure_reaction_relaxation=float(self.config["pressure_reaction_relaxation"]),
            ),
            boundary_model=self.boundary_model,
        )
        self.solid_solver = SolverVBD(
            self.model,
            iterations=int(self.config["rigid_iterations"]),
            integrate_particles=False,
            fsi_boundary_model=self.boundary_model,
        )

        coupling_mode = getattr(self.args, "coupling_mode", "interlinked")
        mode = (
            SolverFSI.Config.CouplingMode.INTERLINKED
            if coupling_mode == "interlinked"
            else SolverFSI.Config.CouplingMode.LOOSE
        )
        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
            config=SolverFSI.Config(
                mode=mode,
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 2))),
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_piston_x = float(self.state_0.body_q.numpy()[self.piston_body, 0])
        self.max_body_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0

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
        self.domain_wire_starts, self.domain_wire_ends = build_box_wireframe(
            min_x=float(self.domain_min[0]),
            max_x=float(self.domain_max[0]),
            min_y=float(self.domain_min[1]),
            max_y=float(self.domain_max[1]),
            min_z=float(self.domain_min[2]),
            max_z=float(self.domain_max[2]),
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.55, 0.42, 0.62),
            pitch=-18.0,
            yaw=-122.0,
        )

        self.reset()
        self.capture()

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def _add_domain_walls(self, builder: newton.ModelBuilder) -> None:
        wall_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_particle_collision=True,
            has_shape_collision=False,
        )
        wall_t = self.wall_thickness
        min_x = float(self.domain_min[0])
        min_y = float(self.domain_min[1])
        min_z = float(self.domain_min[2])
        max_x = float(self.domain_max[0])
        max_y = float(self.domain_max[1])
        max_z = float(self.domain_max[2])
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        cz = 0.5 * (min_z + max_z)
        hx = 0.5 * (max_x - min_x)
        hy = 0.5 * (max_y - min_y)
        hz = 0.5 * (max_z - min_z)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, min_y - wall_t, cz), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, max_y + wall_t, cz), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((min_x - wall_t, cy, cz), wp.quat_identity()),
            hx=wall_t,
            hy=hy + wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, cy, min_z - wall_t), wp.quat_identity()),
            hx=hx + wall_t,
            hy=hy + wall_t,
            hz=wall_t,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, cy, max_z + wall_t), wp.quat_identity()),
            hx=hx + wall_t,
            hy=hy + wall_t,
            hz=wall_t,
            cfg=wall_cfg,
        )

    def reset(self):
        self.sim_time = 0.0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()
        self.max_body_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.viewer._paused = True

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _record_diagnostics(self):
        body_force = self.boundary_model.body_force.numpy()
        sample_force = self.boundary_model.sample_force.numpy()
        density = self.state_0.ipbf.density.numpy()

        if body_force.size:
            self.max_body_force_norm = max(self.max_body_force_norm, float(np.linalg.norm(body_force, axis=1).max()))
        if sample_force.size:
            self.max_sample_force_norm = max(
                self.max_sample_force_norm,
                float(np.linalg.norm(sample_force, axis=1).max()),
            )
        if density.size:
            self.max_density = max(self.max_density, float(np.max(density)))

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
        self._record_diagnostics()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        piston_x = float(body_q[self.piston_body, 0])
        piston_dx = piston_x - self.initial_piston_x
        pressure_relaxation = float(self.config["pressure_reaction_relaxation"])

        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        if pressure_relaxation > 0.0:
            assert self.max_sample_force_norm > 0.0, "pressure reaction did not write sample forces"
            assert self.max_body_force_norm > 0.0, "pressure reaction did not write body forces"
            assert abs(piston_dx) > 1.0e-5, f"dynamic piston was not pushed by pressure reaction: dx={piston_dx:.6f}"
        else:
            assert self.max_sample_force_norm <= 1.0e-8, "pressure reaction should be disabled"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/pressure_domain",
            self.domain_wire_starts,
            self.domain_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.008,
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
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
