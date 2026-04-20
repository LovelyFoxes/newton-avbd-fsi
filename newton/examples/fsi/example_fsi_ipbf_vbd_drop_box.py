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

import argparse

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
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used by the drop-box validation scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=2,
            help="Number of outer IPBF/AVBD feedback passes used in the drop-box pool scene.",
        )
        parser.add_argument(
            "--ipbf-iterations",
            type=int,
            default=None,
            help="Override the number of IPBF density iterations used in the pool scene.",
        )
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=None,
            help="Override the number of simulation substeps per rendered frame.",
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
            help="Scale applied to pressure-gradient reaction forces.",
        )
        parser.add_argument(
            "--include-static-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Include sampled static walls in the FSI boundary density and local dissipation paths.",
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--static-boundary-weight",
            type=float,
            default=None,
            help="Diagnostic weight applied to static boundary samples.",
        )
        parser.add_argument(
            "--viscosity-boundary-coefficient",
            type=float,
            default=None,
            help="Boundary-sample viscosity coefficient used for local wall/body dissipation.",
        )
        parser.add_argument(
            "--viscosity-coefficient",
            type=float,
            default=None,
            help="Fluid-fluid viscosity coefficient used for internal velocity diffusion.",
        )
        parser.add_argument(
            "--xsph-boundary-coefficient",
            type=float,
            default=None,
            help="Boundary-sample XSPH coefficient used for local wall/body velocity smoothing.",
        )
        parser.add_argument(
            "--xsph-coefficient",
            type=float,
            default=None,
            help="Fluid-fluid XSPH velocity smoothing coefficient.",
        )
        parser.add_argument(
            "--boundary-velocity-damping",
            type=float,
            default=None,
            help="Tangential velocity damping applied at final particle-shape contacts.",
        )
        parser.add_argument(
            "--velocity-damping",
            type=float,
            default=None,
            help="Example-level global velocity damping multiplier.",
        )
        parser.add_argument(
            "--box-bottom-gap",
            type=float,
            default=None,
            help="Override the initial gap between the water surface and the box bottom [m].",
        )
        parser.add_argument(
            "--box-center-x",
            type=float,
            default=None,
            help="Override the initial box center x position [m].",
        )
        parser.add_argument(
            "--box-center-z",
            type=float,
            default=None,
            help="Override the initial box center z position [m].",
        )
        parser.add_argument(
            "--enable-runtime-diagnostics",
            action="store_true",
            help="Enable per-frame host-side diagnostics in interactive runs.",
        )
        parser.add_argument(
            "--use-cuda-graph",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Capture simulation launches in a CUDA graph. Disabled by default "
                "for this scene because dynamic fluid-solid contact and boundary "
                "updates need fresh launches for correct behavior."
            ),
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        projection_relaxation = getattr(self.args, "projection_reaction_relaxation", None)
        velocity_reaction_relaxation = getattr(self.args, "velocity_reaction_relaxation", None)
        pressure_relaxation = getattr(self.args, "pressure_reaction_relaxation", None)
        ipbf_iterations_override = getattr(self.args, "ipbf_iterations", None)
        sim_substeps_override = getattr(self.args, "sim_substeps", None)
        box_bottom_gap_override = getattr(self.args, "box_bottom_gap", None)
        box_center_x_override = getattr(self.args, "box_center_x", None)
        box_center_z_override = getattr(self.args, "box_center_z", None)
        include_static_override = getattr(self.args, "include_static_boundary_samples", None)

        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.44,
                "container_half_depth": 0.44,
                "wall_half_height": 0.72,
                "pool_dim_x": 14,
                "pool_dim_y": 5,
                "pool_dim_z": 14,
                "cell": 0.04,
                "mass": 0.064,
                "radius_mean": 0.016,
                "smoothing_radius": 0.075,
                "pool_bottom_clearance": 0.03,
                "box_half_extent": wp.vec3(0.10, 0.055, 0.10),
                "box_center_x": 0.0,
                "box_center_z": 0.0,
                "box_bottom_gap": 0.09,
                "box_density": 1400.0,
                "rest_density": 1000.0,
                "ipbf_iterations": 6,
                "ipbf_relaxation": 0.5,
                "compliance": 1.0e-5,
                "sim_substeps": 6,
                "velocity_damping": 0.996,
                "viscosity_coefficient": 0.002,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.004,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "rigid_iterations": 6,
                "boundary_spacing": 0.04,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "projection_reaction_relaxation": (
                    0.15 if projection_relaxation is None else float(projection_relaxation)
                ),
                "velocity_reaction_relaxation": (
                    0.08 if velocity_reaction_relaxation is None else float(velocity_reaction_relaxation)
                ),
                "pressure_reaction_relaxation": (0.8 if pressure_relaxation is None else float(pressure_relaxation)),
                "expected_min_box_drop": 0.035,
                "expected_min_surface_rise": 0.01,
                "expected_min_body_force_norm": 0.1,
                "expected_min_sample_force_norm": 0.01,
            }
        else:
            config = {
                "container_half_width": 0.60,
                "container_half_depth": 0.60,
                "wall_half_height": 0.92,
                "pool_dim_x": 72,
                "pool_dim_y": 10,
                "pool_dim_z": 72,
                "cell": 0.014,
                "mass": 0.002744,
                "radius_mean": 0.006,
                "smoothing_radius": 0.025,
                "pool_bottom_clearance": 0.03,
                "box_half_extent": wp.vec3(0.09, 0.06, 0.09),
                "box_center_x": 0.0,
                "box_center_z": 0.0,
                "box_bottom_gap": 0.12,
                "box_density": 1600.0,
                "rest_density": 1000.0,
                "ipbf_iterations": 2,
                "ipbf_relaxation": 0.5,
                "compliance": 1.0e-5,
                "sim_substeps": 6,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "viscosity_boundary_coefficient": 0.0,
                "xsph_coefficient": 0.005,
                "xsph_boundary_coefficient": 0.0,
                "boundary_velocity_damping": 1.0,
                "rigid_iterations": 2,
                "boundary_spacing": 0.014,
                "static_boundary_weight": 1.0,
                "include_static_boundary_samples": False,
                "projection_reaction_relaxation": (
                    0.15 if projection_relaxation is None else float(projection_relaxation)
                ),
                "velocity_reaction_relaxation": (
                    0.16 if velocity_reaction_relaxation is None else float(velocity_reaction_relaxation)
                ),
                "pressure_reaction_relaxation": (0.6 if pressure_relaxation is None else float(pressure_relaxation)),
                "expected_min_box_drop": 0.0,
                "expected_min_surface_rise": 0.0,
                "expected_min_body_force_norm": 0.0,
                "expected_min_sample_force_norm": 0.0,
            }

        if ipbf_iterations_override is not None:
            config["ipbf_iterations"] = int(ipbf_iterations_override)
        if sim_substeps_override is not None:
            config["sim_substeps"] = int(sim_substeps_override)
        if box_bottom_gap_override is not None:
            config["box_bottom_gap"] = float(box_bottom_gap_override)
        if box_center_x_override is not None:
            config["box_center_x"] = float(box_center_x_override)
        if box_center_z_override is not None:
            config["box_center_z"] = float(box_center_z_override)
        if include_static_override is not None:
            config["include_static_boundary_samples"] = bool(include_static_override)

        optional_float_overrides = {
            "velocity_damping": getattr(self.args, "velocity_damping", None),
            "viscosity_coefficient": getattr(self.args, "viscosity_coefficient", None),
            "viscosity_boundary_coefficient": getattr(self.args, "viscosity_boundary_coefficient", None),
            "xsph_coefficient": getattr(self.args, "xsph_coefficient", None),
            "xsph_boundary_coefficient": getattr(self.args, "xsph_boundary_coefficient", None),
            "boundary_velocity_damping": getattr(self.args, "boundary_velocity_damping", None),
            "static_boundary_weight": getattr(self.args, "static_boundary_weight", None),
        }
        for key, value in optional_float_overrides.items():
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
        self._reset_key_prev = False
        self.use_cuda_graph = bool(getattr(self.args, "use_cuda_graph", False))

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
        self.include_static_boundary_samples = bool(self.config["include_static_boundary_samples"])
        self.show_boundary_samples = bool(getattr(self.args, "show_boundary_samples", False))
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
            include_static=self.include_static_boundary_samples,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=float(self.config["rest_density"]),
                smoothing_radius=float(self.config["smoothing_radius"]),
                compliance=float(self.config["compliance"]),
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=float(self.config["ipbf_relaxation"]),
                viscosity_coefficient=float(self.config["viscosity_coefficient"]),
                viscosity_boundary_coefficient=float(self.config["viscosity_boundary_coefficient"]),
                xsph_coefficient=float(self.config["xsph_coefficient"]),
                xsph_boundary_coefficient=float(self.config["xsph_boundary_coefficient"]),
                boundary_velocity_damping=float(self.config["boundary_velocity_damping"]),
                fsi_projection_reaction_relaxation=float(self.config["projection_reaction_relaxation"]),
                fsi_velocity_projection_reaction_relaxation=float(self.config["velocity_reaction_relaxation"]),
                fsi_pressure_reaction_relaxation=float(self.config["pressure_reaction_relaxation"]),
                fsi_static_boundary_weight=float(self.config["static_boundary_weight"]),
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
                mode=(
                    SolverFSI.Config.CouplingMode.INTERLINKED
                    if getattr(self.args, "coupling_mode", "interlinked") == "interlinked"
                    else SolverFSI.Config.CouplingMode.LOOSE
                ),
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 2))),
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
        self.box_color = wp.array([wp.vec3(1.0, 0.68, 0.24)], dtype=wp.vec3, device=self.model.device)
        self.rigid_material = wp.array([wp.vec4(0.45, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
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
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_box_linear_speed = 0.0
        self.max_box_angular_speed = 0.0
        self.states_remain_finite = True
        self.enable_runtime_diagnostics = bool(getattr(self.args, "enable_runtime_diagnostics", False)) or bool(
            getattr(self.args, "test", False)
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(2.05, 1.45, 2.25),
            pitch=-20.0,
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
        box_center = wp.vec3(float(self.config["box_center_x"]), box_center_y, float(self.config["box_center_z"]))

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
        ui.text(f"Static boundary samples: {'on' if self.include_static_boundary_samples else 'off'}")
        ui.text(f"Render boundary samples: {'on' if self.show_boundary_samples else 'off'}")

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

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
        self.max_density_ratio = 0.0
        self.max_particle_speed = 0.0
        self.max_box_linear_speed = 0.0
        self.max_box_angular_speed = 0.0
        self.states_remain_finite = True
        self.viewer._paused = True

    def capture(self):
        # Keep graph capture opt-in here: this scene rebuilds collision/boundary
        # state as the box enters the pool, and replaying one captured step can
        # hide or amplify those state changes.
        if self.use_cuda_graph and wp.get_device().is_cuda:
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
        body_qd = self.state_0.body_qd.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        particle_radius = self.model.particle_radius.numpy()

        box_y = float(body_q[self.box_body, 1])
        self.min_box_y = min(self.min_box_y, box_y)
        self.max_particle_y = max(self.max_particle_y, float(np.max(particle_q[:, 1] + particle_radius)))
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(body_q).all()
            and np.isfinite(body_qd).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
            and np.isfinite(density).all()
        )

        if body_force.size:
            self.max_body_force_norm = max(self.max_body_force_norm, float(np.linalg.norm(body_force, axis=1).max()))
        if sample_force.size:
            self.max_sample_force_norm = max(
                self.max_sample_force_norm,
                float(np.linalg.norm(sample_force, axis=1).max()),
            )
        if density.size:
            self.max_density = max(self.max_density, float(np.max(density)))
            self.max_density_ratio = max(
                self.max_density_ratio,
                float(np.max(density) / max(float(self.config["rest_density"]), 1.0e-8)),
            )
        if particle_qd.size:
            self.max_particle_speed = max(self.max_particle_speed, float(np.linalg.norm(particle_qd, axis=1).max()))
        if body_qd.size:
            self.max_box_linear_speed = max(
                self.max_box_linear_speed,
                float(np.linalg.norm(body_qd[:, :3], axis=1).max()),
            )
            self.max_box_angular_speed = max(
                self.max_box_angular_speed,
                float(np.linalg.norm(body_qd[:, 3:], axis=1).max()),
            )

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
        expected_min_box_drop = float(self.config["expected_min_box_drop"])
        expected_min_surface_rise = float(self.config["expected_min_surface_rise"])
        expected_min_body_force_norm = float(self.config["expected_min_body_force_norm"])
        expected_min_sample_force_norm = float(self.config["expected_min_sample_force_norm"])

        assert self.states_remain_finite, "simulation produced non-finite particle/body state"
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_density_ratio < 2.5, f"peak density ratio is too large: rho/rho0={self.max_density_ratio:.3f}"
        assert self.max_particle_speed < 25.0, f"particle speed blew up: vmax={self.max_particle_speed:.3f}"
        assert self.max_box_linear_speed < 20.0, f"box linear speed blew up: vmax={self.max_box_linear_speed:.3f}"
        assert self.max_box_angular_speed < 80.0, f"box angular speed blew up: wmax={self.max_box_angular_speed:.3f}"
        assert self.max_body_force_norm > expected_min_body_force_norm, "FSI body reaction was never accumulated"
        assert self.max_sample_force_norm > expected_min_sample_force_norm, (
            "boundary sample forces were never accumulated"
        )
        assert self.min_box_y < self.initial_box_y - expected_min_box_drop, (
            f"drop box did not descend enough: min_y={self.min_box_y:.4f}, initial_y={self.initial_box_y:.4f}"
        )
        assert self.max_particle_y > self.initial_particle_max_y + expected_min_surface_rise, (
            f"fluid surface did not respond enough: max_y={self.max_particle_y:.4f}, "
            f"initial={self.initial_particle_max_y:.4f}"
        )
        assert box_y < self.initial_box_y - max(0.5 * expected_min_box_drop, 0.01), (
            f"drop box final height barely changed: y={box_y:.4f}"
        )
        assert min_particle_y >= -0.02, f"particles penetrated the floor too much: y={min_particle_y:.4f}"
        assert max_x <= self.container_half_width + 0.05, f"particles escaped along x: {max_x:.4f}"
        assert max_z <= self.container_half_depth + 0.05, f"particles escaped along z: {max_z:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/drop_box_pool_wireframe",
            self.pool_wire_starts,
            self.pool_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_shapes(
            "/fsi/drop_box_rigid_body",
            newton.GeoType.BOX,
            tuple(float(self.box_half_extent[i]) for i in range(3)),
            self._body_xforms([self.box_body]),
            self.box_color,
            self.rigid_material,
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
            hidden=not (self.viewer.show_particles and self.show_boundary_samples),
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
