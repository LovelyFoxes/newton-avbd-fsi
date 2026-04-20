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
# Example FSI IPBF VBD Cloth Patch
#
# A minimal cloth-fluid coupling demo. An IPBF particle block moves into a
# VBD cloth curtain. The shared FSIBoundaryModel samples the cloth triangles,
# IPBF projects fluid particles against them, and VBD consumes the resulting
# cloth-side vertex forces.
#
# Command: python -m newton.examples fsi_ipbf_vbd_cloth_patch
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
    """Visual diagnostic for IPBF particles coupled to a VBD cloth patch."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["loose", "interlinked"],
            default="interlinked",
            help="FSI scheduler mode used for the cloth-fluid demo.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of IPBF/VBD iteration pairs for interlinked mode.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, float | int | wp.vec3]:
        if bool(getattr(self.args, "test", False)):
            return {
                "fluid_dim_x": 4,
                "fluid_dim_y": 3,
                "fluid_dim_z": 3,
                "fluid_cell": 0.030,
                "fluid_radius": 0.012,
                "fluid_mass": 0.018,
                "fluid_center": wp.vec3(0.060, 0.170, 0.0),
                "fluid_velocity": wp.vec3(1.0, 0.0, 0.0),
                "cloth_dim_x": 4,
                "cloth_dim_y": 4,
                "cloth_cell": 0.050,
                "cloth_particle_mass": 0.035,
                "cloth_particle_radius": 0.010,
                "cloth_x": 0.140,
                "cloth_center_y": 0.170,
                "cloth_center_z": 0.0,
                "cloth_tri_ke": 55.0,
                "cloth_tri_ka": 55.0,
                "cloth_tri_kd": 0.15,
                "cloth_edge_ke": 6.0,
                "cloth_edge_kd": 0.04,
                "smoothing_radius": 0.075,
                "rest_density": 1000.0,
                "ipbf_iterations": 2,
                "vbd_iterations": 4,
                "velocity_damping": 0.997,
            }

        return {
            "fluid_dim_x": 9,
            "fluid_dim_y": 6,
            "fluid_dim_z": 5,
            "fluid_cell": 0.025,
            "fluid_radius": 0.010,
            "fluid_mass": 0.011,
            "fluid_center": wp.vec3(-0.040, 0.200, 0.0),
            "fluid_velocity": wp.vec3(0.9, 0.0, 0.0),
            "cloth_dim_x": 10,
            "cloth_dim_y": 8,
            "cloth_cell": 0.045,
            "cloth_particle_mass": 0.030,
            "cloth_particle_radius": 0.009,
            "cloth_x": 0.160,
            "cloth_center_y": 0.200,
            "cloth_center_z": 0.0,
            "cloth_tri_ke": 70.0,
            "cloth_tri_ka": 70.0,
            "cloth_tri_kd": 0.25,
            "cloth_edge_ke": 8.0,
            "cloth_edge_kd": 0.05,
            "smoothing_radius": 0.080,
            "rest_density": 1000.0,
            "ipbf_iterations": 4,
            "vbd_iterations": 6,
            "velocity_damping": 0.998,
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

        self.domain_min = wp.vec3(-0.22, 0.02, -0.32)
        self.domain_max = wp.vec3(0.34, 0.42, 0.32)
        self._build_model()
        self._build_solvers()
        self._build_visualization_data()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.70, 0.38, 0.58),
            pitch=-15.0,
            yaw=-126.0,
        )

        self.reset()
        self.capture()

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

    def _build_model(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        fluid_dim_x = int(self.config["fluid_dim_x"])
        fluid_dim_y = int(self.config["fluid_dim_y"])
        fluid_dim_z = int(self.config["fluid_dim_z"])
        fluid_cell = float(self.config["fluid_cell"])
        fluid_center = self.config["fluid_center"]
        fluid_origin = get_particle_grid_origin_from_center(
            center=(float(fluid_center[0]), float(fluid_center[1]), float(fluid_center[2])),
            dim_x=fluid_dim_x,
            dim_y=fluid_dim_y,
            dim_z=fluid_dim_z,
            cell_x=fluid_cell,
            cell_y=fluid_cell,
            cell_z=fluid_cell,
        )
        builder.add_particle_grid(
            pos=fluid_origin,
            rot=wp.quat_identity(),
            vel=self.config["fluid_velocity"],
            dim_x=fluid_dim_x,
            dim_y=fluid_dim_y,
            dim_z=fluid_dim_z,
            cell_x=fluid_cell,
            cell_y=fluid_cell,
            cell_z=fluid_cell,
            mass=float(self.config["fluid_mass"]),
            jitter=0.0,
            radius_mean=float(self.config["fluid_radius"]),
        )

        self.fluid_particle_start = 0
        self.fluid_particle_count = fluid_dim_x * fluid_dim_y * fluid_dim_z
        self.cloth_particle_start = self.fluid_particle_count

        cloth_dim_x = int(self.config["cloth_dim_x"])
        cloth_dim_y = int(self.config["cloth_dim_y"])
        cloth_cell = float(self.config["cloth_cell"])
        cloth_width = cloth_dim_x * cloth_cell
        cloth_height = cloth_dim_y * cloth_cell
        cloth_pos = wp.vec3(
            float(self.config["cloth_x"]),
            float(self.config["cloth_center_y"]) - 0.5 * cloth_height,
            float(self.config["cloth_center_z"]) + 0.5 * cloth_width,
        )
        cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi * 0.5)
        builder.add_cloth_grid(
            pos=cloth_pos,
            rot=cloth_rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cloth_dim_x,
            dim_y=cloth_dim_y,
            cell_x=cloth_cell,
            cell_y=cloth_cell,
            mass=float(self.config["cloth_particle_mass"]),
            fix_top=True,
            tri_ke=float(self.config["cloth_tri_ke"]),
            tri_ka=float(self.config["cloth_tri_ka"]),
            tri_kd=float(self.config["cloth_tri_kd"]),
            edge_ke=float(self.config["cloth_edge_ke"]),
            edge_kd=float(self.config["cloth_edge_kd"]),
            particle_radius=float(self.config["cloth_particle_radius"]),
        )
        self.cloth_particle_count = (cloth_dim_x + 1) * (cloth_dim_y + 1)

        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

    def _build_solvers(self) -> None:
        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["cloth_cell"]),
            support_radius=float(self.config["smoothing_radius"]),
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=float(self.config["fluid_radius"]) * 2.0,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=float(self.config["rest_density"]),
                smoothing_radius=float(self.config["smoothing_radius"]),
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=0.45,
                viscosity_coefficient=0.0004,
                xsph_coefficient=0.002,
                fsi_pressure_reaction_relaxation=0.15,
                fsi_triangle_contact_enabled=True,
                fsi_triangle_contact_margin=0.0,
                fsi_triangle_contact_relaxation=1.0,
                fluid_particle_start=self.fluid_particle_start,
                fluid_particle_count=self.fluid_particle_count,
            ),
            boundary_model=self.boundary_model,
        )
        self.solid_solver = SolverVBD(
            self.model,
            iterations=int(self.config["vbd_iterations"]),
            particle_start=self.cloth_particle_start,
            particle_count=self.cloth_particle_count,
            fsi_boundary_model=self.boundary_model,
        )

        coupling_mode = getattr(self.args, "coupling_mode", "loose")
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
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 3))),
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

    def _build_visualization_data(self) -> None:
        particle_colors = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        particle_colors[: self.fluid_particle_count] = np.array([0.10, 0.55, 1.00], dtype=np.float32)
        particle_colors[self.cloth_particle_start :] = np.array([0.95, 0.82, 0.22], dtype=np.float32)
        self.particle_colors = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)

        self.boundary_colors = wp.full(
            self.boundary_model.sample_count,
            value=wp.vec3(1.0, 0.45, 0.18),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["cloth_cell"]) * 0.12,
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
            include_top=True,
            device=self.model.device,
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
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()
        self.max_cloth_dx = 0.0
        self.max_triangle_contact_particle_delta = 0.0
        self.max_triangle_contact_vertex_delta = 0.0
        self.max_triangle_contact_pair_count = 0
        self.max_vertex_force_norm = 0.0
        self.max_density = 0.0
        self.viewer._paused = True

    def capture(self):
        # Keep this diagnostic scene uncaptured so Python-side state swaps and
        # per-frame validation metrics always observe the same state object.
        self.graph = None

    def _accumulate_substep_diagnostics(self) -> None:
        triangle_particle_delta = self.fluid_solver._triangle_contact_particle_delta.numpy()
        triangle_vertex_delta = self.boundary_model.vertex_contact_delta.numpy()[self.cloth_particle_start :]
        triangle_pair_count = int(self.fluid_solver._triangle_contact_pair_count.numpy()[0])
        vertex_force = self.boundary_model.vertex_force.numpy()[self.cloth_particle_start :]
        density = self.solver._fluid_state.ipbf.density.numpy()

        self.max_triangle_contact_particle_delta = max(
            self.max_triangle_contact_particle_delta,
            float(np.linalg.norm(triangle_particle_delta, axis=1).max()),
        )
        self.max_triangle_contact_vertex_delta = max(
            self.max_triangle_contact_vertex_delta,
            float(np.linalg.norm(triangle_vertex_delta, axis=1).max()),
        )
        self.max_triangle_contact_pair_count = max(self.max_triangle_contact_pair_count, triangle_pair_count)
        self.max_vertex_force_norm = max(self.max_vertex_force_norm, float(np.linalg.norm(vertex_force, axis=1).max()))
        self.max_density = max(self.max_density, float(np.max(density)))

    def _record_frame_diagnostics(self):
        particle_q = self.state_0.particle_q.numpy()
        cloth_q = particle_q[self.cloth_particle_start :]
        cloth_q0 = self.initial_particle_q[self.cloth_particle_start :]
        self.max_cloth_dx = max(self.max_cloth_dx, float(np.max(cloth_q[:, 0] - cloth_q0[:, 0])))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                scale_velocities,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, float(self.config["velocity_damping"])],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
            self._accumulate_substep_diagnostics()
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
        self._record_frame_diagnostics()
        self.sim_time += self.frame_dt

    def test_final(self):
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_triangle_contact_vertex_delta > 0.0, "cloth triangle-contact deltas were never accumulated"
        assert self.max_vertex_force_norm > 0.0, "cloth-side FSI vertex forces were not accumulated"
        assert self.max_cloth_dx > 1.0e-4, f"cloth patch did not move in the expected +x direction: {self.max_cloth_dx}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/cloth_fluid_domain",
            self.domain_wire_starts,
            self.domain_wire_ends,
            colors=(0.82, 0.86, 0.90),
            width=0.008,
        )
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/fsi/particles_colored",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
        )
        self.viewer.log_points(
            "/fsi/cloth_boundary_samples",
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
