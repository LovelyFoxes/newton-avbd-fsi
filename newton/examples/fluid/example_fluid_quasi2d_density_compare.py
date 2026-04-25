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
# Example Fluid Quasi-2D Density Compare
#
# A tall, thin water column meant to approximate a 2D density-compression
# comparison while still running in the 3D engine. The geometry is intentionally
# thin in depth so cutaway rendering can reveal the density field clearly.
#
# Command: python -m newton.examples fluid_quasi2d_density_compare
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
    color_particles_from_density_comparison,
    create_fluid_solver,
    register_fluid_solver_attributes,
    shared_shape_contact_settings,
)
from newton.examples.ipbf.common import (
    build_box_wireframe,
    get_particle_grid_origin_from_center,
    scale_velocities,
)


class Example:
    """Quasi-2D density-column comparison for IPBF and PBF."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_fluid_solver_argument(parser)
        add_shared_fluid_tuning_arguments(parser)
        parser.add_argument(
            "--box-clamp",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Clamp particles to the tall comparison tank after each substep.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.14,
                "container_half_depth": 0.040,
                "wall_half_height": 0.42,
                "pool_dim_x": 16,
                "pool_dim_y": 36,
                "pool_dim_z": 4,
                "cell": 0.020,
                "mass": 0.008,
                "radius": 0.0085,
                "smoothing_radius": 0.038,
                "pool_bottom_clearance": 0.012,
                "iterations": 4,
                "sim_substeps": 5,
                "velocity_damping": 0.998,
                "viscosity": 0.0015,
                "xsph": 0.0025,
                "rest_density": 1000.0,
                "compliance": 0.0,
                "ipbf_relaxation": 0.5,
                "pbf_relaxation": 1.0,
                "lambda_regularization": 1.0e-6,
                "use_constraint_clamp": True,
            }
            return apply_shared_fluid_tuning_overrides(self.args, config)

        config = {
            "container_half_width": 0.18,
            "container_half_depth": 0.050,
            "wall_half_height": 0.68,
            "pool_dim_x": 28,
            "pool_dim_y": 80,
            "pool_dim_z": 6,
            "cell": 0.012,
            "mass": 0.001728,
            "radius": 0.0052,
            "smoothing_radius": 0.022,
            "pool_bottom_clearance": 0.012,
            "iterations": 8,
            "sim_substeps": 4,
            "velocity_damping": 0.999,
            "viscosity": 0.0012,
            "xsph": 0.0020,
            "rest_density": 1000.0,
            "compliance": 0.0,
            "ipbf_relaxation": 0.5,
            "pbf_relaxation": 1.0,
            "lambda_regularization": 1.0e-6,
            "use_constraint_clamp": True,
        }
        return apply_shared_fluid_tuning_overrides(self.args, config)

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self.fluid_solver_name = str(getattr(self.args, "fluid_solver", "ipbf"))
        self.use_box_clamp = bool(getattr(self.args, "box_clamp", True))

        scene = self._get_scene_config()
        contact_settings = shared_shape_contact_settings(particle_radius=float(scene["radius"]))
        self.scene = scene
        self.sim_substeps = int(scene["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(scene["container_half_width"])
        self.container_half_depth = float(scene["container_half_depth"])
        self.wall_half_height = float(scene["wall_half_height"])
        self.wall_thickness = float(contact_settings["wall_thickness"])
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(scene["velocity_damping"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        register_fluid_solver_attributes(builder, self.fluid_solver_name)

        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.margin = float(contact_settings["shape_margin"])
        builder.default_shape_cfg.gap = float(contact_settings["shape_gap"])
        self._add_container(builder)
        self._add_pool(builder)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))
        self.solver = create_fluid_solver(
            model=self.model,
            fluid_solver_name=self.fluid_solver_name,
            scene=scene,
            boundary_model=None,
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
            value=wp.vec3(0.08, 0.22, 0.92),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.particle_radii = wp.full(
            self.model.particle_count,
            value=float(scene["radius"]) * 0.78,
            dtype=wp.float32,
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
            include_top=False,
        )

        self.max_density_ratio = 0.0
        self.max_density_ratio_rms = 0.0
        self.states_remain_finite = True

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.0, self.top_y * 0.54, 1.65),
            pitch=-4.0,
            yaw=-90.0,
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

    def _add_pool(self, builder: newton.ModelBuilder) -> None:
        pool_span_y = float(self.scene["cell"]) * float(self.scene["pool_dim_y"])
        center = (
            0.0,
            self.floor_y + float(self.scene["pool_bottom_clearance"]) + 0.5 * pool_span_y,
            0.0,
        )
        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=center,
                dim_x=int(self.scene["pool_dim_x"]),
                dim_y=int(self.scene["pool_dim_y"]),
                dim_z=int(self.scene["pool_dim_z"]),
                cell_x=float(self.scene["cell"]),
                cell_y=float(self.scene["cell"]),
                cell_z=float(self.scene["cell"]),
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(self.scene["pool_dim_x"]),
            dim_y=int(self.scene["pool_dim_y"]),
            dim_z=int(self.scene["pool_dim_z"]),
            cell_x=float(self.scene["cell"]),
            cell_y=float(self.scene["cell"]),
            cell_z=float(self.scene["cell"]),
            mass=float(self.scene["mass"]),
            jitter=0.0,
            radius_mean=float(self.scene["radius"]),
        )

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Fluid solver: {self.fluid_solver_name.upper()}")
        ui.text("Scene: quasi-2D density column")
        ui.text(f"Box clamp: {'on' if self.use_box_clamp else 'off'}")
        ui.text(f"Max density ratio: {self.max_density_ratio:.3f}")
        ui.text(f"Max density RMS: {self.max_density_ratio_rms:.4f}")

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self.contacts.clear()
        self.max_density_ratio = 0.0
        self.max_density_ratio_rms = 0.0
        self.states_remain_finite = True
        self._update_particle_colors()

    def capture(self):
        """Compatibility hook for examples API; this scene does not pre-capture graphs."""

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, control=None, contacts=self.contacts, dt=self.sim_dt)
            if self.use_box_clamp:
                wp.launch(
                    clamp_particles_to_box,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_1.particle_q,
                        self.state_1.particle_qd,
                        self.model.particle_radius,
                        self.container_half_width,
                        self.floor_y,
                        self.top_y + 100.0,
                        self.container_half_depth,
                    ],
                    device=self.model.device,
                )
            if self.velocity_damping != 1.0:
                wp.launch(
                    scale_velocities,
                    dim=self.model.particle_count,
                    inputs=[self.state_1.particle_qd, self.velocity_damping],
                    device=self.model.device,
                )

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _fluid_state_namespace(self, state: newton.State):
        return state.ipbf if self.fluid_solver_name == "ipbf" else state.pbf

    def _update_particle_colors(self) -> None:
        density = self._fluid_state_namespace(self.state_0).density
        wp.launch(
            color_particles_from_density_comparison,
            dim=self.model.particle_count,
            inputs=[density, float(self.scene["rest_density"])],
            outputs=[self.particle_colors],
            device=self.model.device,
        )

    def _record_diagnostics(self) -> None:
        density = self._fluid_state_namespace(self.state_0).density.numpy()
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(density).all() or not np.isfinite(q).all():
            self.states_remain_finite = False
            return

        density_ratio = density / max(float(self.scene["rest_density"]), 1.0e-8)
        self.max_density_ratio = max(self.max_density_ratio, float(np.max(density_ratio)))
        self.max_density_ratio_rms = max(
            self.max_density_ratio_rms,
            float(np.sqrt(np.mean((density_ratio - 1.0) * (density_ratio - 1.0)))),
        )

    def step(self):
        self.simulate()
        self._update_particle_colors()
        self._record_diagnostics()

    def test_final(self):
        assert self.states_remain_finite, "quasi-2D density column produced non-finite state"
        assert self.max_density_ratio > 0.0, "density diagnostics were not updated"
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q)), "particle positions became non-finite"
        if self.use_box_clamp:
            assert float(np.max(np.abs(q[:, 0]))) <= self.container_half_width + 0.02
            assert float(np.max(np.abs(q[:, 2]))) <= self.container_half_depth + 0.02

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/container_wire",
            self.box_wire_starts,
            self.box_wire_ends,
            colors=(0.88, 0.90, 0.95),
        )
        self.viewer.log_points(
            "/fluid_particles",
            points=self.state_0.particle_q,
            colors=self.particle_colors,
            radii=self.particle_radii,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
