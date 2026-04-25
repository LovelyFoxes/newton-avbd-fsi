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
# Example Fluid Quasi-2D Box Compare
#
# A tall, thin water column with one flattened rigid box resting on the free
# surface. The scene is designed as a quasi-2D FSI density-comparison setup
# for IPBF+AVBD and PBF+AVBD. Use ``--box-case float`` or ``--box-case sink``
# to switch between a light box and a heavy box that sinks into the column.
#
# Command: python -m newton.examples fluid_quasi2d_box_compare
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
    clamp_particles_to_box,
    color_particles_from_density_comparison,
    register_fluid_solver_attributes,
)
from newton.examples.ipbf.common import (
    build_box_wireframe,
    get_particle_grid_half_span,
    get_particle_grid_origin_from_center,
    scale_velocities,
)
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverPBF, SolverVBD


class Example:
    """Quasi-2D rigid FSI density comparison for IPBF and PBF."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        add_fluid_solver_argument(parser)
        add_shared_fluid_tuning_arguments(parser)
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used by the quasi-2D box comparison scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of outer fluid-solid feedback passes used per simulation substep.",
        )
        parser.add_argument(
            "--box-case",
            choices=["float", "sink"],
            default="float",
            help="Choose a light floating box or a heavier sinking box preset.",
        )
        parser.add_argument(
            "--box-density",
            type=float,
            default=None,
            help="Override the rigid box density [kg/m^3].",
        )
        parser.add_argument(
            "--projection-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to particle-shape projection reaction forces.",
        )
        parser.add_argument(
            "--velocity-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to boundary velocity projection reaction forces.",
        )
        parser.add_argument(
            "--pressure-reaction-relaxation",
            type=float,
            default=None,
            help="Scale applied to boundary pressure-reaction forces.",
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--box-clamp",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Clamp particles to the tall comparison tank after each substep.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        fluid_iterations_override = getattr(self.args, "fluid_iterations", None)
        sim_substeps_override = getattr(self.args, "sim_substeps", None)
        box_density_override = getattr(self.args, "box_density", None)
        box_case = str(getattr(self.args, "box_case", "float"))

        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.15,
                "container_half_depth": 0.040,
                "wall_half_height": 0.44,
                "pool_dim_x": 16,
                "pool_dim_y": 30,
                "pool_dim_z": 4,
                "cell": 0.020,
                "mass": 0.008,
                "radius_mean": 0.0085,
                "smoothing_radius": 0.038,
                "pool_bottom_clearance": 0.012,
                "box_half_extent": wp.vec3(0.080, 0.016, 0.018),
                "box_bottom_gap": 0.006,
                "box_density_float": 240.0,
                "box_density_sink": 1600.0,
                "rest_density": 1000.0,
                "fluid_iterations": 5,
                "sim_substeps": 6,
                "velocity_damping": 0.998,
                "viscosity_coefficient": 0.0020,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.0030,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 0.995,
                "rigid_iterations": 5,
                "boundary_spacing": 0.020,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "projection_reaction_relaxation": 0.18,
                "velocity_reaction_relaxation": 0.08,
                "pressure_reaction_relaxation": 1.0,
                "pbf_relaxation": 0.8,
                "lambda_regularization": 5.0e-6,
            }
        else:
            config = {
                "container_half_width": 0.16,
                "container_half_depth": 0.040,
                "wall_half_height": 0.74,
                "pool_dim_x": 24,
                "pool_dim_y": 70,
                "pool_dim_z": 4,
                "cell": 0.012,
                "mass": 0.001728,
                "radius_mean": 0.0052,
                "smoothing_radius": 0.022,
                "pool_bottom_clearance": 0.012,
                "box_half_extent": wp.vec3(0.090, 0.024, 0.018),
                "box_bottom_gap": 0.004,
                "box_density_float": 240.0,
                "box_density_sink": 1500.0,
                "rest_density": 1000.0,
                "fluid_iterations": 8,
                "sim_substeps": 4,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0018,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.0025,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 0.995,
                "rigid_iterations": 8,
                "boundary_spacing": 0.012,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "projection_reaction_relaxation": 0.18,
                "velocity_reaction_relaxation": 0.10,
                "pressure_reaction_relaxation": 1.0,
                "pbf_relaxation": 0.5,
                "lambda_regularization": 5.0e-6,
            }

        if fluid_iterations_override is not None:
            config["fluid_iterations"] = int(fluid_iterations_override)
        if sim_substeps_override is not None:
            config["sim_substeps"] = int(sim_substeps_override)
        if box_density_override is not None:
            config["box_density"] = float(box_density_override)
        else:
            config["box_density"] = float(config["box_density_float" if box_case == "float" else "box_density_sink"])

        optional_overrides = {
            "projection_reaction_relaxation": getattr(self.args, "projection_reaction_relaxation", None),
            "velocity_reaction_relaxation": getattr(self.args, "velocity_reaction_relaxation", None),
            "pressure_reaction_relaxation": getattr(self.args, "pressure_reaction_relaxation", None),
        }
        for key, value in optional_overrides.items():
            if value is not None:
                config[key] = float(value)

        return config

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self.fluid_solver_name = str(getattr(self.args, "fluid_solver", "ipbf"))
        self.box_case = str(getattr(self.args, "box_case", "float"))
        self.show_boundary_samples = bool(getattr(self.args, "show_boundary_samples", False))
        self.use_box_clamp = bool(getattr(self.args, "box_clamp", True))
        self.config = self._get_scene_config()
        self.include_static_boundary_samples = bool(self.config["include_static_boundary_samples"])
        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(self.config["container_half_width"])
        self.container_half_depth = float(self.config["container_half_depth"])
        self.wall_half_height = float(self.config["wall_half_height"])
        self.wall_thickness = 0.04
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(self.config["velocity_damping"])
        self.box_half_extent = wp.vec3(self.config["box_half_extent"])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        self._register_fluid_attributes(builder)
        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)
        self.float_box_body = self._add_float_box(builder)
        self._add_pool(builder)
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))
        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            include_static=self.include_static_boundary_samples,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = self._create_fluid_solver()
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
                mode=(
                    SolverFSI.Config.CouplingMode.INTERLINKED
                    if getattr(self.args, "coupling_mode", "interlinked") == "interlinked"
                    else SolverFSI.Config.CouplingMode.LOOSE
                ),
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
            value=wp.vec3(0.08, 0.22, 0.92),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.particle_radii = wp.full(
            self.model.particle_count,
            value=float(self.config["radius_mean"]) * 0.78,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.boundary_colors = self._build_boundary_colors()
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * 0.18,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.float_box_color = wp.array([wp.vec3(0.82, 0.72, 0.18)], dtype=wp.vec3, device=self.model.device)
        self.rigid_material = wp.array([wp.vec4(0.45, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.pool_wire_starts, self.pool_wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=self.container_half_width,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
            include_top=True,
        )

        self.initial_box_y = 0.0
        self.min_box_y = 0.0
        self.max_density_ratio = 0.0
        self.max_density_ratio_rms = 0.0
        self.states_remain_finite = True

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.0, self.top_y * 0.56, 1.65),
            pitch=-4.0,
            yaw=-90.0,
        )

        self.reset()

    def _register_fluid_attributes(self, builder: newton.ModelBuilder) -> None:
        register_fluid_solver_attributes(builder, self.fluid_solver_name)

    def _create_fluid_solver(self):
        common_kwargs = {
            "rest_density": float(self.config["rest_density"]),
            "smoothing_radius": float(self.config["smoothing_radius"]),
            "iterations": int(self.config["fluid_iterations"]),
            "viscosity_coefficient": float(self.config["viscosity_coefficient"]),
            "viscosity_boundary_coefficient": float(self.config["viscosity_boundary_coefficient"]),
            "xsph_coefficient": float(self.config["xsph_coefficient"]),
            "xsph_boundary_coefficient": float(self.config["xsph_boundary_coefficient"]),
            "boundary_velocity_damping": float(self.config["boundary_velocity_damping"]),
            "fsi_projection_reaction_relaxation": float(self.config["projection_reaction_relaxation"]),
            "fsi_velocity_projection_reaction_relaxation": float(self.config["velocity_reaction_relaxation"]),
            "fsi_pressure_reaction_relaxation": float(self.config["pressure_reaction_relaxation"]),
            "fsi_static_boundary_weight": float(self.config["static_boundary_weight"]),
        }
        if self.fluid_solver_name == "ipbf":
            return SolverIPBF(
                self.model,
                SolverIPBF.Config(
                    **common_kwargs,
                    kernel_family=SolverIPBF.Config.KernelFamily.CUBIC_SPLINE,
                    compliance=1.0e-5,
                    relaxation=0.5,
                    use_constraint_clamp=True,
                    fluid_particle_start=0,
                    fluid_particle_count=None,
                ),
                boundary_model=self.boundary_model,
            )

        return SolverPBF(
            self.model,
                SolverPBF.Config(
                    **common_kwargs,
                    kernel_family=SolverPBF.Config.KernelFamily.CUBIC_SPLINE,
                    relaxation=float(self.config.get("pbf_relaxation", 1.0)),
                    lambda_regularization=float(self.config.get("lambda_regularization", 1.0e-6)),
                    use_constraint_clamp=True,
                    fluid_particle_start=0,
                    fluid_particle_count=None,
                ),
            boundary_model=self.boundary_model,
        )

    def _fluid_state_namespace(self, state: newton.State):
        return state.ipbf if self.fluid_solver_name == "ipbf" else state.pbf

    def _build_boundary_colors(self) -> wp.array[wp.vec3]:
        sample_body = self.boundary_model.sample_body.numpy()
        colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        colors[sample_body == self.float_box_body] = np.array([0.82, 0.72, 0.18], dtype=np.float32)
        colors[sample_body < 0] = np.array([0.86, 0.90, 0.95], dtype=np.float32)
        return wp.array(colors, dtype=wp.vec3, device=self.model.device)

    def _body_xforms(self, body_indices: list[int]) -> wp.array[wp.transform]:
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

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
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 2.0 * hy + wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
            cfg=wall_cfg,
        )

    def _compute_pool_surface_y(self) -> float:
        _, pool_half_span_y, _ = get_particle_grid_half_span(
            dim_x=int(self.config["pool_dim_x"]),
            dim_y=int(self.config["pool_dim_y"]),
            dim_z=int(self.config["pool_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
        )
        return (
            self.floor_y
            + float(self.config["pool_bottom_clearance"])
            + 2.0 * pool_half_span_y
            + float(self.config["radius_mean"])
        )

    def _add_float_box(self, builder: newton.ModelBuilder) -> int:
        water_surface_y = self._compute_pool_surface_y()
        hy = float(self.box_half_extent[1])
        box_center_y = water_surface_y + float(self.config["box_bottom_gap"]) + hy
        box_center = wp.vec3(0.0, box_center_y, 0.0)

        box_body = builder.add_body(
            xform=wp.transform(box_center, wp.quat_identity()),
            label="quasi2d_compare_box",
        )
        builder.add_shape_box(
            body=box_body,
            hx=float(self.box_half_extent[0]),
            hy=hy,
            hz=float(self.box_half_extent[2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=float(self.config["box_density"]),
                mu=0.0,
            ),
        )
        return box_body

    def _add_pool(self, builder: newton.ModelBuilder) -> None:
        pool_center = (
            0.0,
            self.floor_y
            + float(self.config["pool_bottom_clearance"])
            + get_particle_grid_half_span(
                dim_x=int(self.config["pool_dim_x"]),
                dim_y=int(self.config["pool_dim_y"]),
                dim_z=int(self.config["pool_dim_z"]),
                cell_x=float(self.config["cell"]),
                cell_y=float(self.config["cell"]),
                cell_z=float(self.config["cell"]),
            )[1],
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
        ui.text(f"Fluid solver: {self.fluid_solver_name.upper()}")
        ui.text(f"Box case: {self.box_case}")
        ui.text(f"Box density: {float(self.config['box_density']):.1f} kg/m^3")
        ui.text(f"Box clamp: {'on' if self.use_box_clamp else 'off'}")
        ui.text(f"Max density ratio: {self.max_density_ratio:.3f}")
        ui.text(f"Max density RMS: {self.max_density_ratio_rms:.4f}")
        ui.text(f"Boundary samples: {'on' if self.show_boundary_samples else 'off'}")

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
        self.initial_box_y = float(body_q[self.float_box_body, 1])
        self.min_box_y = self.initial_box_y
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
                        self.top_y,
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

    def _update_particle_colors(self) -> None:
        density = self._fluid_state_namespace(self.state_0).density
        wp.launch(
            color_particles_from_density_comparison,
            dim=self.model.particle_count,
            inputs=[density, float(self.config["rest_density"])],
            outputs=[self.particle_colors],
            device=self.model.device,
        )

    def _record_diagnostics(self) -> None:
        density = self._fluid_state_namespace(self.state_0).density.numpy()
        q = self.state_0.particle_q.numpy()
        box_q = self.state_0.body_q.numpy()

        if not np.isfinite(density).all() or not np.isfinite(q).all() or not np.isfinite(box_q).all():
            self.states_remain_finite = False
            return

        density_ratio = density / max(float(self.config["rest_density"]), 1.0e-8)
        self.max_density_ratio = max(self.max_density_ratio, float(np.max(density_ratio)))
        self.max_density_ratio_rms = max(
            self.max_density_ratio_rms,
            float(np.sqrt(np.mean((density_ratio - 1.0) * (density_ratio - 1.0)))),
        )
        self.min_box_y = min(self.min_box_y, float(box_q[self.float_box_body, 1]))

    def step(self):
        self.simulate()
        self._update_particle_colors()
        self._record_diagnostics()

    def test_final(self):
        assert self.states_remain_finite, "quasi-2D box comparison produced non-finite state"
        assert self.max_density_ratio > 0.0, "density diagnostics were not updated"
        q = self.state_0.particle_q.numpy()
        if self.use_box_clamp:
            assert float(np.max(np.abs(q[:, 0]))) <= self.container_half_width + 0.02
            assert float(np.max(np.abs(q[:, 2]))) <= self.container_half_depth + 0.02

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/container_wire",
            self.pool_wire_starts,
            self.pool_wire_ends,
            colors=(0.88, 0.90, 0.95),
        )
        self.viewer.log_shapes(
            "/fsi/quasi2d_compare_box",
            newton.GeoType.BOX,
            tuple(float(self.box_half_extent[i]) for i in range(3)),
            self._body_xforms([self.float_box_body]),
            self.float_box_color,
            self.rigid_material,
        )
        self.viewer.log_points(
            "/fluid_particles",
            points=self.state_0.particle_q,
            colors=self.particle_colors,
            radii=self.particle_radii,
        )
        if self.show_boundary_samples and self.boundary_model.sample_count > 0:
            self.viewer.log_points(
                "/boundary_samples",
                points=self.boundary_model.sample_x_world,
                colors=self.boundary_colors,
                radii=self.boundary_radii,
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
