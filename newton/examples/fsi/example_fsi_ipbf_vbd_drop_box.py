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
# Example FSI IPBF VBD Drop Box
#
# An open-top pool scene for AVBD/IPBF fluid-solid interaction. A flat IPBF
# water slab rests at the bottom of a container while a dynamic AVBD box drops
# into the pool under gravity. Particle-shape projection and boundary-density
# pressure reaction both contribute to the body motion.
#
# Command: python -m newton.examples fsi_ipbf_vbd_drop_box
#
###########################################################################

from __future__ import annotations

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
    """Open-top pool drop test for AVBD/IPBF fluid-solid coupling."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of IPBF/AVBD iteration pairs used in the drop-box pool scene.",
        )
        parser.add_argument(
            "--projection-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to particle-shape projection reaction forces.",
        )
        parser.add_argument(
            "--pressure-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to pressure-gradient reaction forces.",
        )
        parser.add_argument(
            "--enable-runtime-diagnostics",
            action="store_true",
            help="Enable per-frame host-side diagnostics in interactive runs.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        projection_relaxation = getattr(self.args, "projection_reaction_relaxation", None)
        pressure_relaxation = getattr(self.args, "pressure_reaction_relaxation", None)

        if bool(getattr(self.args, "test", False)):
            return {
                "container_half_width": 0.34,
                "container_half_depth": 0.34,
                "wall_half_height": 0.48,
                "pool_dim_x": 8,
                "pool_dim_y": 3,
                "pool_dim_z": 8,
                "cell": 0.05,
                "mass": 0.11,
                "radius_mean": 0.02,
                "smoothing_radius": 0.09,
                "pool_bottom_clearance": 0.03,
                "box_half_extent": wp.vec3(0.08, 0.05, 0.08),
                "box_bottom_gap": 0.08,
                "box_density": 650.0,
                "ipbf_iterations": 5,
                "ipbf_relaxation": 0.5,
                "compliance": 1.0e-5,
                "sim_substeps": 6,
                "velocity_damping": 0.994,
                "viscosity": 0.002,
                "xsph": 0.004,
                "rigid_iterations": 2,
                "boundary_spacing": 0.05,
                "projection_reaction_relaxation": (
                    0.004 if projection_relaxation is None else float(projection_relaxation)
                ),
                "pressure_reaction_relaxation": (0.08 if pressure_relaxation is None else float(pressure_relaxation)),
            }

        return {
            "container_half_width": 0.34,
            "container_half_depth": 0.34,
            "wall_half_height": 0.48,
            "pool_dim_x": 8,
            "pool_dim_y": 3,
            "pool_dim_z": 8,
            "cell": 0.05,
            "mass": 0.11,
            "radius_mean": 0.02,
            "smoothing_radius": 0.09,
            "pool_bottom_clearance": 0.03,
            "box_half_extent": wp.vec3(0.08, 0.05, 0.08),
            "box_bottom_gap": 0.08,
            "box_density": 650.0,
            "ipbf_iterations": 5,
            "ipbf_relaxation": 0.5,
            "compliance": 1.0e-5,
            "sim_substeps": 6,
            "velocity_damping": 0.994,
            "viscosity": 0.002,
            "xsph": 0.004,
            "rigid_iterations": 2,
            "boundary_spacing": 0.05,
            "projection_reaction_relaxation": (
                0.004 if projection_relaxation is None else float(projection_relaxation)
            ),
            "pressure_reaction_relaxation": (0.08 if pressure_relaxation is None else float(pressure_relaxation)),
        }

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False

        self.config = self._get_scene_config()
        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(self.config["container_half_width"])
        self.container_half_depth = float(self.config["container_half_depth"])
        self.wall_half_height = float(self.config["wall_half_height"])
        self.wall_thickness = 0.05
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(self.config["velocity_damping"])
        self.box_half_extent = self.config["box_half_extent"]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0

        self._add_container(builder)
        self.box_body = self._add_dynamic_box(builder)
        self._add_pool(builder)
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

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
                rest_density=1000.0,
                smoothing_radius=float(self.config["smoothing_radius"]),
                compliance=float(self.config["compliance"]),
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=float(self.config["ipbf_relaxation"]),
                viscosity_coefficient=float(self.config["viscosity"]),
                xsph_coefficient=float(self.config["xsph"]),
                fsi_projection_reaction_relaxation=float(self.config["projection_reaction_relaxation"]),
                fsi_velocity_projection_reaction_relaxation=float(self.config["projection_reaction_relaxation"]),
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

        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
            config=SolverFSI.Config(
                mode=SolverFSI.Config.CouplingMode.INTERLINKED,
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
            value=float(self.config["boundary_spacing"]) * 0.16,
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

        self.initial_box_y = 0.0
        self.min_box_y = 0.0
        self.initial_particle_max_y = 0.0
        self.max_particle_y = 0.0
        self.max_body_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.enable_runtime_diagnostics = bool(getattr(self.args, "enable_runtime_diagnostics", False)) or bool(
            getattr(self.args, "test", False)
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(1.35, 1.0, 1.45),
            pitch=-18.0,
            yaw=-132.0,
        )

        self.reset()
        self.capture()

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

    def _add_dynamic_box(self, builder: newton.ModelBuilder) -> int:
        _, pool_half_span_y, _ = get_particle_grid_half_span(
            dim_x=int(self.config["pool_dim_x"]),
            dim_y=int(self.config["pool_dim_y"]),
            dim_z=int(self.config["pool_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
        )
        pool_top_surface_y = (
            self.floor_y
            + float(self.config["pool_bottom_clearance"])
            + 2.0 * pool_half_span_y
            + float(self.config["radius_mean"])
        )
        box_center_y = pool_top_surface_y + float(self.config["box_bottom_gap"]) + float(self.box_half_extent[1])
        box_center = wp.vec3(0.0, box_center_y, 0.0)

        box_body = builder.add_body(
            xform=wp.transform(box_center, wp.quat_identity()),
            label="fsi_drop_box",
        )
        builder.add_shape_box(
            body=box_body,
            hx=float(self.box_half_extent[0]),
            hy=float(self.box_half_extent[1]),
            hz=float(self.box_half_extent[2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=float(self.config["box_density"]),
                mu=0.0,
            ),
        )
        return box_body

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
        self.initial_box_y = float(body_q[self.box_body, 1])
        self.min_box_y = self.initial_box_y
        self.initial_particle_max_y = float(np.max(particle_q[:, 1] + particle_radius))
        self.max_particle_y = self.initial_particle_max_y
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

    def _record_diagnostics(self) -> None:
        body_force = self.boundary_model.body_force.numpy()
        sample_force = self.boundary_model.sample_force.numpy()
        density = self.state_0.ipbf.density.numpy()
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()

        box_y = float(body_q[self.box_body, 1])
        self.min_box_y = min(self.min_box_y, box_y)
        self.max_particle_y = max(self.max_particle_y, float(np.max(particle_q[:, 1] + particle_radius)))

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
        if self.enable_runtime_diagnostics:
            self._record_diagnostics()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        box_y = float(self.state_0.body_q.numpy()[self.box_body, 1])

        max_x = float(np.max(np.abs(particle_q[:, 0]) + particle_radius))
        max_z = float(np.max(np.abs(particle_q[:, 2]) + particle_radius))
        min_particle_y = float(np.min(particle_q[:, 1] - particle_radius))

        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_body_force_norm > 0.0, "FSI body reaction was never accumulated"
        assert self.max_sample_force_norm > 0.0, "boundary sample forces were never accumulated"
        assert self.min_box_y < self.initial_box_y - 0.04, (
            f"drop box did not descend enough: min_y={self.min_box_y:.4f}, initial_y={self.initial_box_y:.4f}"
        )
        assert self.max_particle_y > self.initial_particle_max_y + 0.01, (
            f"fluid surface did not respond enough: max_y={self.max_particle_y:.4f}, "
            f"initial={self.initial_particle_max_y:.4f}"
        )
        assert box_y < self.initial_box_y - 0.01, f"drop box final height barely changed: y={box_y:.4f}"
        assert min_particle_y >= -0.02, f"particles penetrated the floor too much: y={min_particle_y:.4f}"
        assert max_x <= self.container_half_width + 0.08, f"particles escaped along x: {max_x:.4f}"
        assert max_z <= self.container_half_depth + 0.08, f"particles escaped along z: {max_z:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/drop_box_pool_wireframe",
            self.pool_wire_starts,
            self.pool_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_points(
            "/fsi/drop_box_ipbf_particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/drop_box_boundary_samples",
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
