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
# Example FSI IPBF VBD Box Wall Break
#
# Large-tank AVBD/IPBF scene where a gated fluid block in the left third of the
# tank is released to impact a brick-stacked wall of many dynamic rigid boxes
# near the middle/right partition. The box wall cycles three densities and uses
# interlinked fluid-solid iterations by default.
#
# Command: python -m newton.examples fsi_ipbf_vbd_box_wall_break
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


def _centered_line_positions(count: int, spacing: float) -> list[float]:
    if count <= 0:
        return []
    return [spacing * (float(i) - 0.5 * float(count - 1)) for i in range(count)]


class Example:
    """Gate-release scene where a fluid column impacts a dense box wall."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used for the box-wall scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of outer IPBF/AVBD feedback passes used per simulation substep.",
        )
        parser.add_argument(
            "--release-step",
            type=int,
            default=None,
            help="Frame step at which the fluid gate starts opening.",
        )
        parser.add_argument(
            "--gate-open-duration-frames",
            type=int,
            default=None,
            help="Number of frames used to retract the fluid gate after release.",
        )
        parser.add_argument(
            "--interlinked-switch-delay-frames",
            type=int,
            default=None,
            help=(
                "Delay after gate release before switching from loose settling to interlinked coupling. "
                "Only used when --coupling-mode interlinked."
            ),
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--enable-runtime-diagnostics",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Record per-frame diagnostics while the example runs.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            config = {
                "container_half_width": 0.72,
                "container_half_depth": 0.30,
                "wall_half_height": 0.44,
                "wall_thickness": 0.035,
                "fluid_bottom_clearance": 0.03,
                "fluid_dim_x": 14,
                "fluid_dim_y": 16,
                "fluid_dim_z": 10,
                "cell": 0.032,
                "mass": 0.032768,
                "radius_mean": 0.012,
                "smoothing_radius": 0.075,
                "gate_half_thickness": 0.018,
                "gate_open_height": 1.60,
                "gate_open_duration_frames": 6,
                "release_step": 20,
                "interlinked_switch_delay_frames": 8,
                "box_half_extents": (0.040, 0.040, 0.040),
                "box_gap_x": 0.006,
                "box_gap_y": 0.006,
                "box_gap_z": 0.006,
                "box_floor_clearance": 0.003,
                "box_layers_x": 1,
                "box_layers_y": 5,
                "box_columns_z": 5,
                "box_densities": (300.0, 700.0, 1100.0),
                "box_mu": 0.15,
                "rest_density": 1000.0,
                "ipbf_iterations": 6,
                "rigid_iterations": 6,
                "sim_substeps": 5,
                "velocity_damping": 0.997,
                "viscosity_coefficient": 0.0020,
                "xsph_coefficient": 0.004,
                "boundary_velocity_damping": 1.0,
                "fsi_pressure_reaction_relaxation": 1.0,
                "static_boundary_weight": 0.5,
                "boundary_spacing": 0.032,
                "water_render_radius_scale": 0.48,
                "boundary_render_radius_scale": 0.30,
                "shape_contact_ke": 3.0e4,
                "shape_contact_kd": 0.1,
                "shape_contact_gap": 0.002,
                "expected_min_box_dx": 0.004,
                "expected_min_body_force_norm": 0.01,
                "expected_min_fluid_front_x": -0.02,
            }
        else:
            config = {
                "container_half_width": 1.15,
                "container_half_depth": 0.50,
                "wall_half_height": 0.70,
                "wall_thickness": 0.040,
                "fluid_bottom_clearance": 0.04,
                "fluid_dim_x": 28,
                "fluid_dim_y": 28,
                "fluid_dim_z": 18,
                "cell": 0.022,
                "mass": 0.010648,
                "radius_mean": 0.0082,
                "smoothing_radius": 0.040,
                "gate_half_thickness": 0.020,
                "gate_open_height": 2.40,
                "gate_open_duration_frames": 8,
                "release_step": 30,
                "interlinked_switch_delay_frames": 20,
                "box_half_extents": (0.038, 0.038, 0.038),
                "box_gap_x": 0.006,
                "box_gap_y": 0.006,
                "box_gap_z": 0.006,
                "box_floor_clearance": 0.003,
                "box_layers_x": 2,
                "box_layers_y": 7,
                "box_columns_z": 7,
                "box_densities": (250.0, 700.0, 1250.0),
                "box_mu": 0.15,
                "rest_density": 1000.0,
                "ipbf_iterations": 6,
                "rigid_iterations": 6,
                "sim_substeps": 5,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "xsph_coefficient": 0.006,
                "boundary_velocity_damping": 0.97,
                "fsi_pressure_reaction_relaxation": 1.5,
                "static_boundary_weight": 0.20,
                "boundary_spacing": 0.022,
                "water_render_radius_scale": 0.48,
                "boundary_render_radius_scale": 0.30,
                "shape_contact_ke": 4.0e4,
                "shape_contact_kd": 0.1,
                "shape_contact_gap": 0.002,
                "expected_min_box_dx": 0.0,
                "expected_min_body_force_norm": 0.0,
                "expected_min_fluid_front_x": 0.0,
            }

        release_step = getattr(self.args, "release_step", None)
        if release_step is not None:
            config["release_step"] = max(0, int(release_step))

        gate_open_duration_frames = getattr(self.args, "gate_open_duration_frames", None)
        if gate_open_duration_frames is not None:
            config["gate_open_duration_frames"] = max(1, int(gate_open_duration_frames))

        interlinked_switch_delay_frames = getattr(self.args, "interlinked_switch_delay_frames", None)
        if interlinked_switch_delay_frames is not None:
            config["interlinked_switch_delay_frames"] = max(0, int(interlinked_switch_delay_frames))

        return config

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index = 0
        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False
        self.show_boundary_samples = bool(getattr(self.args, "show_boundary_samples", False))
        self.config = self._get_scene_config()
        self.enable_runtime_diagnostics = bool(
            getattr(self.args, "enable_runtime_diagnostics", False) or bool(getattr(self.args, "test", False))
        )

        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.container_half_width = float(self.config["container_half_width"])
        self.container_half_depth = float(self.config["container_half_depth"])
        self.wall_half_height = float(self.config["wall_half_height"])
        self.wall_thickness = float(self.config["wall_thickness"])
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.velocity_damping = float(self.config["velocity_damping"])
        self.box_half_extents = tuple(float(v) for v in self.config["box_half_extents"])
        self.box_palette = (
            np.array([1.0, 0.78, 0.24], dtype=np.float32),
            np.array([0.34, 0.86, 0.34], dtype=np.float32),
            np.array([0.20, 0.92, 0.92], dtype=np.float32),
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.rigid_gap = float(self.config["shape_contact_gap"])
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.ke = float(self.config["shape_contact_ke"])
        builder.default_shape_cfg.kd = float(self.config["shape_contact_kd"])
        builder.default_shape_cfg.gap = float(self.config["shape_contact_gap"])

        self._add_container(builder)
        self._add_fluid_gate(builder)
        self.wall_box_bodies = self._add_box_wall(builder)
        self._add_fluid_block(builder)
        builder.color()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))
        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE,
            include_static=True,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=float(self.config["rest_density"]),
                smoothing_radius=float(self.config["smoothing_radius"]),
                compliance=1.0e-5,
                iterations=int(self.config["ipbf_iterations"]),
                relaxation=0.5,
                viscosity_coefficient=float(self.config["viscosity_coefficient"]),
                xsph_coefficient=float(self.config["xsph_coefficient"]),
                boundary_velocity_damping=float(self.config["boundary_velocity_damping"]),
                fsi_pressure_reaction_relaxation=float(self.config["fsi_pressure_reaction_relaxation"]),
                fsi_static_boundary_weight=float(self.config["static_boundary_weight"]),
                fluid_particle_start=self.fluid_particle_start,
                fluid_particle_count=self.fluid_particle_count,
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
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 3))),
            ),
        )
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            self.args,
            rigid_contact_max=max(10000, len(self.wall_box_bodies) * 160),
            soft_contact_margin=float(self.config["radius_mean"]) * 2.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        self.particle_colors = wp.full(
            self.fluid_particle_count,
            value=wp.vec3(0.48, 0.82, 1.0),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.water_radii = wp.full(
            self.fluid_particle_count,
            value=float(self.config["radius_mean"]) * float(self.config["water_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.box_colors = wp.array(self.box_colors_np, dtype=wp.vec3, device=self.model.device)
        self.gate_colors = wp.array([wp.vec3(0.82, 0.84, 0.88)], dtype=wp.vec3, device=self.model.device)
        self.rigid_material = wp.array([wp.vec4(0.42, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.boundary_colors = self._build_boundary_colors()
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * float(self.config["boundary_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.tank_wire_starts, self.tank_wire_ends = build_box_wireframe(
            min_x=-self.container_half_width,
            max_x=self.container_half_width,
            min_y=self.floor_y,
            max_y=self.top_y,
            min_z=-self.container_half_depth,
            max_z=self.container_half_depth,
            device=self.model.device,
            include_top=False,
        )

        self.initial_box_x = np.zeros(len(self.wall_box_bodies), dtype=np.float32)
        self.max_box_positive_dx = 0.0
        self.max_fluid_front_x = 0.0
        self.max_body_force_norm = 0.0
        self.max_body_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.max_particle_speed = 0.0
        self.max_body_linear_speed = 0.0
        self.max_body_angular_speed = 0.0
        self.states_remain_finite = True

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(2.35, 1.35, 2.15), pitch=-18.0, yaw=-133.0)
        self.reset()

    def _build_boundary_colors(self) -> wp.array(dtype=wp.vec3):
        sample_body = self.boundary_model.sample_body.numpy()
        colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        colors[sample_body < 0] = np.array([0.92, 0.94, 0.98], dtype=np.float32)
        body_color_map = dict(zip(self.wall_box_bodies, self.box_colors_np, strict=True))
        body_color_map[self.gate_body] = np.array([0.82, 0.84, 0.88], dtype=np.float32)
        for body, color in body_color_map.items():
            colors[sample_body == body] = color
        return wp.array(colors, dtype=wp.vec3, device=self.model.device)

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

    def _left_gate_x(self) -> float:
        return -self.container_half_width / 3.0

    def _wall_plane_x(self) -> float:
        return self.container_half_width / 3.0

    def _compute_fluid_center(self) -> tuple[float, float, float]:
        half_span_x, half_span_y, _ = get_particle_grid_half_span(
            dim_x=int(self.config["fluid_dim_x"]),
            dim_y=int(self.config["fluid_dim_y"]),
            dim_z=int(self.config["fluid_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
        )
        section_min_x = -self.container_half_width + self.wall_thickness + half_span_x
        section_max_x = self._left_gate_x() - float(self.config["gate_half_thickness"]) - half_span_x
        center_x = 0.5 * (section_min_x + section_max_x)
        center_y = self.floor_y + float(self.config["fluid_bottom_clearance"]) + half_span_y
        return (center_x, center_y, 0.0)

    def _contact_shape_cfg(self, *, density: float, mu: float) -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(
            density=density,
            ke=float(self.config["shape_contact_ke"]),
            kd=float(self.config["shape_contact_kd"]),
            mu=mu,
            gap=float(self.config["shape_contact_gap"]),
        )

    def _add_container(self, builder: newton.ModelBuilder) -> None:
        wall_cfg = self._contact_shape_cfg(density=0.0, mu=0.0)
        hx = self.container_half_width
        hz = self.container_half_depth
        hy = self.wall_half_height
        wt = self.wall_thickness

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, self.floor_y - wt, 0.0), wp.quat_identity()),
            hx=hx + wt,
            hy=wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((hx + wt, hy, 0.0), wp.quat_identity()),
            hx=wt,
            hy=hy + wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-(hx + wt), hy, 0.0), wp.quat_identity()),
            hx=wt,
            hy=hy + wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, hz + wt), wp.quat_identity()),
            hx=hx,
            hy=hy + wt,
            hz=wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, hy, -(hz + wt)), wp.quat_identity()),
            hx=hx,
            hy=hy + wt,
            hz=wt,
            cfg=wall_cfg,
        )

    def _add_fluid_gate(self, builder: newton.ModelBuilder) -> None:
        gate_half_thickness = float(self.config["gate_half_thickness"])
        self.gate_half_extents = (
            gate_half_thickness,
            self.wall_half_height + self.wall_thickness,
            self.container_half_depth + self.wall_thickness,
        )
        self.gate_closed_center = (self._left_gate_x(), self.wall_half_height, 0.0)
        self.gate_open_center = (
            self.gate_closed_center[0],
            float(self.config["gate_open_height"]),
            self.gate_closed_center[2],
        )
        self.gate_body = builder.add_body(
            xform=wp.transform(self.gate_closed_center, wp.quat_identity()),
            is_kinematic=True,
            label="fsi_box_wall_gate",
        )
        builder.add_shape_box(
            body=self.gate_body,
            hx=self.gate_half_extents[0],
            hy=self.gate_half_extents[1],
            hz=self.gate_half_extents[2],
            cfg=self._contact_shape_cfg(density=0.0, mu=0.0),
        )

    def _box_layer_z_positions(self, layer_y: int) -> list[float]:
        spacing_z = 2.0 * self.box_half_extents[2] + float(self.config["box_gap_z"])
        base_count = int(self.config["box_columns_z"])
        count = base_count if layer_y % 2 == 0 else max(1, base_count - 1)
        return _centered_line_positions(count, spacing_z)

    def _add_box_wall(self, builder: newton.ModelBuilder) -> list[int]:
        spacing_x = 2.0 * self.box_half_extents[0] + float(self.config["box_gap_x"])
        spacing_y = 2.0 * self.box_half_extents[1] + float(self.config["box_gap_y"])
        x_positions = _centered_line_positions(int(self.config["box_layers_x"]), spacing_x)
        densities = tuple(float(v) for v in self.config["box_densities"])
        wall_x = self._wall_plane_x()
        base_y = self.floor_y + self.box_half_extents[1] + float(self.config["box_floor_clearance"])

        self.box_colors_np = np.zeros((0, 3), dtype=np.float32)
        bodies: list[int] = []
        colors: list[np.ndarray] = []
        for layer_x, x_offset in enumerate(x_positions):
            center_x = wall_x + x_offset
            for layer_y in range(int(self.config["box_layers_y"])):
                center_y = base_y + float(layer_y) * spacing_y
                for column_z, center_z in enumerate(self._box_layer_z_positions(layer_y)):
                    density_index = (layer_x + layer_y + column_z) % len(densities)
                    body = builder.add_body(
                        xform=wp.transform((center_x, center_y, center_z), wp.quat_identity()),
                        label=f"fsi_box_wall_{layer_x}_{layer_y}_{column_z}",
                    )
                    builder.add_shape_box(
                        body=body,
                        hx=self.box_half_extents[0],
                        hy=self.box_half_extents[1],
                        hz=self.box_half_extents[2],
                        cfg=self._contact_shape_cfg(density=densities[density_index], mu=float(self.config["box_mu"])),
                    )
                    bodies.append(body)
                    colors.append(self.box_palette[density_index].copy())

        self.box_colors_np = np.asarray(colors, dtype=np.float32)
        return bodies

    def _add_fluid_block(self, builder: newton.ModelBuilder) -> None:
        self.fluid_particle_start = builder.particle_count
        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=self._compute_fluid_center(),
                dim_x=int(self.config["fluid_dim_x"]),
                dim_y=int(self.config["fluid_dim_y"]),
                dim_z=int(self.config["fluid_dim_z"]),
                cell_x=float(self.config["cell"]),
                cell_y=float(self.config["cell"]),
                cell_z=float(self.config["cell"]),
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(self.config["fluid_dim_x"]),
            dim_y=int(self.config["fluid_dim_y"]),
            dim_z=int(self.config["fluid_dim_z"]),
            cell_x=float(self.config["cell"]),
            cell_y=float(self.config["cell"]),
            cell_z=float(self.config["cell"]),
            mass=float(self.config["mass"]),
            jitter=0.0,
            radius_mean=float(self.config["radius_mean"]),
            flags=int(newton.ParticleFlags.ACTIVE),
        )
        self.fluid_particle_count = builder.particle_count - self.fluid_particle_start

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Mode: {getattr(self.args, 'coupling_mode', 'interlinked')}")
        ui.text(f"Frame: {self.frame_index}")
        ui.text(f"Gate release: {int(self.config['release_step'])}")
        ui.text(f"Gate duration: {int(self.config['gate_open_duration_frames'])}")
        ui.text(f"Interlinked switch delay: {int(self.config['interlinked_switch_delay_frames'])}")
        ui.text(f"Box count: {len(self.wall_box_bodies)}")
        ui.text(f"Boundary samples: {'on' if self.show_boundary_samples else 'off'}")

    @staticmethod
    def _lerp_center(
        center_a: tuple[float, float, float],
        center_b: tuple[float, float, float],
        alpha: float,
    ) -> tuple[float, float, float]:
        alpha = max(0.0, min(1.0, alpha))
        return (
            (1.0 - alpha) * center_a[0] + alpha * center_b[0],
            (1.0 - alpha) * center_a[1] + alpha * center_b[1],
            (1.0 - alpha) * center_a[2] + alpha * center_b[2],
        )

    def _gate_alpha(self) -> float:
        release_step = int(self.config["release_step"])
        duration = int(self.config["gate_open_duration_frames"])
        if self.frame_index < release_step:
            return 0.0
        if duration <= 1:
            return 1.0
        return float(self.frame_index - release_step + 1) / float(duration)

    def _set_body_pose(
        self,
        state: newton.State,
        body_idx: int,
        center: tuple[float, float, float],
        rotation: wp.quat,
    ) -> None:
        body_q_np = state.body_q.numpy()
        body_q_np[body_idx, 0] = float(center[0])
        body_q_np[body_idx, 1] = float(center[1])
        body_q_np[body_idx, 2] = float(center[2])
        body_q_np[body_idx, 3:7] = np.array(rotation, dtype=np.float32)
        state.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)

        body_qd_np = state.body_qd.numpy()
        body_qd_np[body_idx, 0:6] = 0.0
        state.body_qd = wp.array(body_qd_np, dtype=wp.spatial_vector, device=self.model.device)

    def _apply_gate_state(self, state: newton.State) -> None:
        center = self._lerp_center(self.gate_closed_center, self.gate_open_center, self._gate_alpha())
        self._set_body_pose(state, self.gate_body, center, wp.quat_identity())

    def _update_solver_coupling_mode(self) -> None:
        if getattr(self.args, "coupling_mode", "interlinked") != "interlinked":
            self.solver.config.mode = SolverFSI.Config.CouplingMode.LOOSE
            return

        switch_step = int(self.config["release_step"]) + int(self.config["interlinked_switch_delay_frames"])
        self.solver.config.mode = (
            SolverFSI.Config.CouplingMode.LOOSE
            if self.frame_index < switch_step
            else SolverFSI.Config.CouplingMode.INTERLINKED
        )

    def reset(self):
        self.sim_time = 0.0
        self.frame_index = 0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self._apply_gate_state(self.state_0)
        self._apply_gate_state(self.state_1)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()

        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        self.initial_box_x = body_q[self.wall_box_bodies, 0].astype(np.float32).copy()
        self.max_box_positive_dx = 0.0
        self.max_fluid_front_x = float(np.max(particle_q[:, 0] + particle_radius))
        self.max_body_force_norm = 0.0
        self.max_body_step_avg_force_norm = 0.0
        self.max_sample_force_norm = 0.0
        self.max_density = 0.0
        self.max_particle_speed = 0.0
        self.max_body_linear_speed = 0.0
        self.max_body_angular_speed = 0.0
        self.states_remain_finite = True
        self.viewer._paused = True

    def _record_diagnostics(self) -> None:
        body_force = self.boundary_model.body_force.numpy()
        body_force_step_avg = self.boundary_model.body_force_step_avg.numpy()
        sample_force = self.boundary_model.sample_force.numpy()
        density = self.state_0.ipbf.density.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        particle_radius = self.model.particle_radius.numpy()

        self.max_box_positive_dx = max(
            self.max_box_positive_dx,
            float(np.max(body_q[self.wall_box_bodies, 0] - self.initial_box_x)),
        )
        self.max_fluid_front_x = max(self.max_fluid_front_x, float(np.max(particle_q[:, 0] + particle_radius)))
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(body_force).all()
            and np.isfinite(body_force_step_avg).all()
            and np.isfinite(sample_force).all()
            and np.isfinite(density).all()
            and np.isfinite(body_q).all()
            and np.isfinite(body_qd).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
        )

        if body_force.size:
            self.max_body_force_norm = max(self.max_body_force_norm, float(np.linalg.norm(body_force, axis=1).max()))
        if body_force_step_avg.size:
            self.max_body_step_avg_force_norm = max(
                self.max_body_step_avg_force_norm,
                float(np.linalg.norm(body_force_step_avg[self.wall_box_bodies], axis=1).max()),
            )
        if sample_force.size:
            self.max_sample_force_norm = max(
                self.max_sample_force_norm,
                float(np.linalg.norm(sample_force, axis=1).max()),
            )
        if density.size:
            self.max_density = max(self.max_density, float(np.max(density)))
        if particle_qd.size:
            self.max_particle_speed = max(self.max_particle_speed, float(np.linalg.norm(particle_qd, axis=1).max()))
        if body_qd.size:
            wall_body_qd = body_qd[self.wall_box_bodies]
            self.max_body_linear_speed = max(
                self.max_body_linear_speed,
                float(np.linalg.norm(wall_body_qd[:, :3], axis=1).max()),
            )
            self.max_body_angular_speed = max(
                self.max_body_angular_speed,
                float(np.linalg.norm(wall_body_qd[:, 3:], axis=1).max()),
            )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._update_solver_coupling_mode()
            self._apply_gate_state(self.state_0)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
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
        if self.enable_runtime_diagnostics:
            self._record_diagnostics()
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def test_final(self):
        if not self.enable_runtime_diagnostics:
            self._record_diagnostics()
        assert self.states_remain_finite, "box-wall scene produced non-finite diagnostics"
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        assert self.max_box_positive_dx > float(self.config["expected_min_box_dx"]), (
            f"box wall did not move enough: {self.max_box_positive_dx}"
        )
        assert max(self.max_body_force_norm, self.max_body_step_avg_force_norm) > float(
            self.config["expected_min_body_force_norm"]
        ), "FSI body reaction stayed too small"
        assert self.max_fluid_front_x > float(self.config["expected_min_fluid_front_x"]), (
            f"released fluid front did not advance enough: {self.max_fluid_front_x}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/fsi/box_wall_break_tank_wire",
            self.tank_wire_starts,
            self.tank_wire_ends,
            colors=(0.88, 0.90, 0.95),
            width=0.010,
        )
        self.viewer.log_shapes(
            "/fsi/box_wall_break_gate",
            newton.GeoType.BOX,
            self.gate_half_extents,
            self._body_xforms([self.gate_body]),
            self.gate_colors,
            self.rigid_material,
        )
        self.viewer.log_shapes(
            "/fsi/box_wall_break_boxes",
            newton.GeoType.BOX,
            self.box_half_extents,
            self._body_xforms(self.wall_box_bodies),
            self.box_colors,
            self.rigid_material,
        )
        self.viewer.log_points(
            "/fsi/box_wall_break_water_points",
            points=wp.array(
                self.state_0.particle_q.numpy()[
                    self.fluid_particle_start : self.fluid_particle_start + self.fluid_particle_count
                ],
                dtype=wp.vec3,
                device=self.model.device,
            ),
            radii=self.water_radii,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/box_wall_break_boundary_samples",
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
