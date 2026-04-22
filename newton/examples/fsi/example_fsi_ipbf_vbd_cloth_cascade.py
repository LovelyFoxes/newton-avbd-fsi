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
# Example FSI IPBF VBD Cloth Cascade
#
# Release-driven cloth-fluid interaction scene. A rectangular reservoir stores
# an IPBF fluid block behind a square outlet gate. After a configurable delay,
# the gate opens and the water flows through a horizontal tunnel onto a
# three-stage VBD cloth cascade before entering a bottom catch tank.
#
# Command: python -m newton.examples fsi_ipbf_vbd_cloth_cascade
#
###########################################################################

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ipbf.common import build_box_wireframe, get_particle_grid_origin_from_center, scale_velocities
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD


def _compose_quats(*quats: wp.quat) -> wp.quat:
    q = wp.quat_identity()
    for quat in quats:
        q = wp.mul(quat, q)
    return q


def _quat_rotate_np(quat: wp.quat, vec: tuple[float, float, float]) -> np.ndarray:
    rot = np.array(wp.quat_to_matrix(quat), dtype=np.float32).reshape(3, 3)
    return rot @ np.array(vec, dtype=np.float32)


class Example:
    """Cloth-fluid cascade scene with delayed fluid-gate release."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["loose", "interlinked"],
            default="interlinked",
            help="FSI scheduler mode used for the cascade scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=3,
            help="Number of outer fluid-solid feedback passes used in interlinked mode.",
        )
        parser.add_argument(
            "--cloth-self-contact",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable cloth-owned VBD self-contact.",
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--triangle-contact-relaxation",
            type=float,
            default=None,
            help="Override the IPBF cloth triangle-contact relaxation.",
        )
        parser.add_argument(
            "--pressure-reaction-relaxation",
            type=float,
            default=None,
            help="Override the IPBF cloth pressure-reaction relaxation.",
        )
        parser.add_argument(
            "--release-step",
            type=int,
            default=None,
            help="Frame step at which the fluid gate opens.",
        )
        parser.add_argument(
            "--fluid-release-step",
            type=int,
            default=None,
            help="Frame step at which the fluid gate opens.",
        )
        parser.add_argument(
            "--gate-open-duration-frames",
            type=int,
            default=None,
            help="Number of frames used to retract the fluid gate after release.",
        )
        parser.add_argument(
            "--fluid-gate-open-duration-frames",
            type=int,
            default=None,
            help="Number of frames used to retract the fluid gate after release.",
        )
        return parser

    def _get_scene_config(self) -> dict[str, object]:
        if bool(getattr(self.args, "test", False)):
            config = {
                "tank_half_width": 0.44,
                "tank_half_depth": 0.30,
                "tank_wall_height": 0.28,
                "tank_wall_thickness": 0.03,
                "fluid_reservoir_center": (-0.58, 1.02, 0.0),
                "fluid_reservoir_half_width": 0.15,
                "fluid_reservoir_half_height": 0.30,
                "fluid_reservoir_half_depth": 0.12,
                "fluid_reservoir_wall_thickness": 0.015,
                "fluid_outlet_half_height": 0.045,
                "fluid_outlet_half_depth": 0.045,
                "tunnel_half_length": 0.28,
                "tunnel_floor_half_thickness": 0.018,
                "tunnel_top_half_thickness": 0.012,
                "tunnel_side_thickness": 0.012,
                "fluid_cell": 0.022,
                "fluid_dim_x": 10,
                "fluid_dim_y": 24,
                "fluid_dim_z": 10,
                "fluid_mass": 0.012,
                "fluid_radius": 0.009,
                "fluid_gate_half_thickness": 0.014,
                "fluid_gate_open_height": 2.4,
                "fluid_gate_open_duration_frames": 6,
                "cloth_dim_x": 6,
                "cloth_dim_y": 6,
                "cloth_cell_x": 0.080,
                "cloth_cell_y": 0.080,
                "cloth_particle_mass": 0.032,
                "cloth_particle_radius": 0.010,
                "cloth_centers": (
                    (0.10, 0.86, 0.0),
                    (-0.12, 0.56, 0.0),
                    (0.12, 0.28, 0.0),
                ),
                "cloth_tilt_degrees": (30.0, -30.0, 22.0),
                "cloth_tri_ke": 95.0,
                "cloth_tri_ka": 95.0,
                "cloth_tri_kd": 0.28,
                "cloth_edge_ke": 10.0,
                "cloth_edge_kd": 0.06,
                "cloth_self_contact_radius": 0.020,
                "cloth_self_contact_margin": 0.034,
                "rest_density": 1000.0,
                "smoothing_radius": 0.075,
                "ipbf_iterations": 6,
                "vbd_iterations": 8,
                "sim_substeps": 5,
                "velocity_damping": 0.997,
                "viscosity_coefficient": 0.0020,
                "xsph_coefficient": 0.004,
                "boundary_velocity_damping": 1.0,
                "fsi_pressure_reaction_relaxation": 1.0,
                "static_boundary_weight": 0.5,
                "triangle_contact_relaxation": 1.0,
                "boundary_spacing": 0.030,
                "water_render_radius_scale": 0.86,
                "shape_contact_ke": 3.0e4,
                "shape_contact_kd": 1.2e3,
                "shape_contact_gap": 0.008,
                "release_step": 20,
                "show_domain_wire": True,
                "expected_min_cloth_dx": 0.0,
                "expected_min_triangle_pairs": 0,
                "expected_min_particles_in_tank": 0,
            }
        else:
            config = {
                "tank_half_width": 1.00,
                "tank_half_depth": 0.80,
                "tank_wall_height": 0.64,
                "tank_wall_thickness": 0.035,
                "fluid_reservoir_center": (-0.68, 1.4, 0.0),
                "fluid_reservoir_half_width": 0.24,
                "fluid_reservoir_half_height": 0.56,
                "fluid_reservoir_half_depth": 0.24,
                "fluid_reservoir_wall_thickness": 0.020,
                "fluid_outlet_half_height": 0.080,
                "fluid_outlet_half_depth": 0.080,
                "tunnel_half_length": 0.18,
                "tunnel_floor_half_thickness": 0.020,
                "tunnel_top_half_thickness": 0.014,
                "tunnel_side_thickness": 0.014,
                "fluid_cell": 0.018,
                "fluid_dim_x": 24,
                "fluid_dim_y": 54,
                "fluid_dim_z": 24,
                "fluid_mass": 0.0058,
                "fluid_radius": 0.0068,
                "fluid_gate_half_thickness": 0.016,
                "fluid_gate_open_height": 2.8,
                "fluid_gate_open_duration_frames": 8,
                "cloth_dim_x": 16,
                "cloth_dim_y": 16,
                "cloth_cell_x": 0.035,
                "cloth_cell_y": 0.035,
                "cloth_particle_mass": 0.0075,
                "cloth_particle_radius": 0.0055,
                "cloth_centers": (
                    (0.20, 0.90, 0.05),
                    (-0.20, 0.65, -0.05),
                    (0.20, 0.40, 0.0),
                ),
                "cloth_tilt_degrees": (35.0, -30.0, 30.0),
                "cloth_tri_ke": 240.0,
                "cloth_tri_ka": 240.0,
                "cloth_tri_kd": 0.24,
                "cloth_edge_ke": 16.0,
                "cloth_edge_kd": 0.06,
                "cloth_self_contact_radius": 0.012,
                "cloth_self_contact_margin": 0.020,
                "rest_density": 1000.0,
                "smoothing_radius": 0.040,
                "ipbf_iterations": 6,
                "vbd_iterations": 6,
                "sim_substeps": 5,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "xsph_coefficient": 0.006,
                "boundary_velocity_damping": 0.97,
                "fsi_pressure_reaction_relaxation": 1.0,
                "static_boundary_weight": 0.20,
                "triangle_contact_relaxation": 1.0,
                "boundary_spacing": 0.020,
                "water_render_radius_scale": 0.86,
                "shape_contact_ke": 4.0e4,
                "shape_contact_kd": 1.5e3,
                "shape_contact_gap": 0.008,
                "release_step": 30,
                "show_domain_wire": True,
                "expected_min_cloth_dx": 0.0,
                "expected_min_triangle_pairs": 0,
                "expected_min_particles_in_tank": 0,
            }

        release_step = getattr(self.args, "release_step", None)
        fluid_release_step = getattr(self.args, "fluid_release_step", None)
        default_release_step = int(config["release_step"])
        config["fluid_release_step"] = (
            default_release_step if fluid_release_step is None else max(0, int(fluid_release_step))
        )
        if release_step is not None:
            config["fluid_release_step"] = max(0, int(release_step))

        gate_open_duration_frames = getattr(self.args, "gate_open_duration_frames", None)
        fluid_gate_open_duration_frames = getattr(self.args, "fluid_gate_open_duration_frames", None)
        if gate_open_duration_frames is not None:
            config["fluid_gate_open_duration_frames"] = max(1, int(gate_open_duration_frames))
        if fluid_gate_open_duration_frames is not None:
            config["fluid_gate_open_duration_frames"] = max(1, int(fluid_gate_open_duration_frames))

        triangle_contact_relaxation = getattr(self.args, "triangle_contact_relaxation", None)
        if triangle_contact_relaxation is not None:
            config["triangle_contact_relaxation"] = float(triangle_contact_relaxation)
        pressure_reaction_relaxation = getattr(self.args, "pressure_reaction_relaxation", None)
        if pressure_reaction_relaxation is not None:
            config["fsi_pressure_reaction_relaxation"] = float(pressure_reaction_relaxation)

        config["cloth_self_contact_enabled"] = bool(getattr(self.args, "cloth_self_contact", False))
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
        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.floor_y = 0.0

        self.domain_min = wp.vec3(-1.00, -0.08, -0.62)
        self.domain_max = wp.vec3(0.95, 1.92, 0.62)

        self._build_model()
        self._build_solvers()
        self._build_visualization_data()

        self.viewer.set_model(self.model)
        self.viewer.show_visual = False
        self.viewer.show_triangles = True
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.35, 1.42, 3.05),
            pitch=-15.0,
            yaw=-94.0,
        )

        self.reset()
        self.capture()

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text("Release-driven cloth-fluid cascade")
        ui.text(f"Mode: {getattr(self.args, 'coupling_mode', 'interlinked')}")
        ui.text(f"Frame: {self.frame_index}")
        ui.text(f"Fluid gate release: {int(self.config['fluid_release_step'])}")
        ui.text(f"Fluid gate duration: {int(self.config['fluid_gate_open_duration_frames'])}")
        ui.text(f"Fluid gate open: {'yes' if self._fluid_gate_is_open() else 'no'}")
        ui.text(f"Pressure reaction: {float(self.config['fsi_pressure_reaction_relaxation']):.2f}")

    def _tunnel_rotation(self) -> wp.quat:
        return wp.quat_identity()

    def _cloth_rotation(self, tilt_degrees: float) -> wp.quat:
        q_flat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)
        q_tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(float(tilt_degrees)))
        return _compose_quats(q_flat, q_tilt)

    def _add_tank(self, builder: newton.ModelBuilder) -> None:
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
        hx = float(self.config["tank_half_width"])
        hz = float(self.config["tank_half_depth"])
        hy = float(self.config["tank_wall_height"])
        wt = float(self.config["tank_wall_thickness"])

        builder.add_ground_plane(height=self.floor_y, cfg=wall_cfg, label="fsi_cascade_ground")

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
            xform=wp.transform((hx + wt, self.floor_y + hy, 0.0), wp.quat_identity()),
            hx=wt,
            hy=hy + wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-(hx + wt), self.floor_y + hy, 0.0), wp.quat_identity()),
            hx=wt,
            hy=hy + wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, self.floor_y + hy, hz + wt), wp.quat_identity()),
            hx=hx,
            hy=hy + wt,
            hz=wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, self.floor_y + hy, -(hz + wt)), wp.quat_identity()),
            hx=hx,
            hy=hy + wt,
            hz=wt,
            cfg=wall_cfg,
        )

    def _add_fluid_reservoir(self, builder: newton.ModelBuilder) -> None:
        self.fluid_reservoir_center = tuple(float(v) for v in self.config["fluid_reservoir_center"])
        self.fluid_reservoir_half_width = float(self.config["fluid_reservoir_half_width"])
        self.fluid_reservoir_half_height = float(self.config["fluid_reservoir_half_height"])
        self.fluid_reservoir_half_depth = float(self.config["fluid_reservoir_half_depth"])
        self.fluid_reservoir_wall_thickness = float(self.config["fluid_reservoir_wall_thickness"])
        self.fluid_outlet_half_height = float(self.config["fluid_outlet_half_height"])
        self.fluid_outlet_half_depth = float(self.config["fluid_outlet_half_depth"])

        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
        cx, cy, cz = self.fluid_reservoir_center
        hx = self.fluid_reservoir_half_width
        hy = self.fluid_reservoir_half_height
        hz = self.fluid_reservoir_half_depth
        wt = self.fluid_reservoir_wall_thickness
        outlet_h = self.fluid_outlet_half_height
        outlet_d = self.fluid_outlet_half_depth
        reservoir_bottom = cy - hy
        reservoir_top = cy + hy
        self.fluid_reservoir_bottom_y = reservoir_bottom
        # The outlet and tunnel are floor-aligned so water drains from the bottom-right corner.
        self.fluid_outlet_center_y = reservoir_bottom + outlet_h
        outlet_y = self.fluid_outlet_center_y

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, cy - hy - wt, cz), wp.quat_identity()),
            hx=hx + wt,
            hy=wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx - hx - wt, cy, cz), wp.quat_identity()),
            hx=wt,
            hy=hy + wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        for z_sign in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform((cx, cy, cz + z_sign * (hz + wt)), wp.quat_identity()),
                hx=hx + wt,
                hy=hy + wt,
                hz=wt,
                cfg=wall_cfg,
            )

        right_wall_x = cx + hx + wt
        self.fluid_outlet_center = (cx + hx, outlet_y, cz)
        self.fluid_outlet_wall_center_x = right_wall_x

        lower_top = outlet_y - outlet_h
        upper_bottom = outlet_y + outlet_h
        if lower_top > reservoir_bottom:
            lower_half_height = 0.5 * (lower_top - reservoir_bottom)
            builder.add_shape_box(
                body=-1,
                xform=wp.transform((right_wall_x, reservoir_bottom + lower_half_height, cz), wp.quat_identity()),
                hx=wt,
                hy=lower_half_height,
                hz=hz + wt,
                cfg=wall_cfg,
            )
        if upper_bottom < reservoir_top:
            upper_half_height = 0.5 * (reservoir_top - upper_bottom)
            builder.add_shape_box(
                body=-1,
                xform=wp.transform((right_wall_x, upper_bottom + upper_half_height, cz), wp.quat_identity()),
                hx=wt,
                hy=upper_half_height,
                hz=hz + wt,
                cfg=wall_cfg,
            )

        side_half_depth = 0.5 * (hz - outlet_d)
        if side_half_depth > 0.0:
            for z_sign in (-1.0, 1.0):
                side_center_z = cz + z_sign * (outlet_d + side_half_depth)
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform((right_wall_x, outlet_y, side_center_z), wp.quat_identity()),
                    hx=wt,
                    hy=outlet_h,
                    hz=side_half_depth,
                    cfg=wall_cfg,
                )

    def _add_fluid_tunnel(self, builder: newton.ModelBuilder) -> None:
        self.tunnel_rotation = self._tunnel_rotation()
        self.tunnel_half_length = float(self.config["tunnel_half_length"])
        self.tunnel_floor_half_thickness = float(self.config["tunnel_floor_half_thickness"])
        self.tunnel_top_half_thickness = float(self.config["tunnel_top_half_thickness"])
        self.tunnel_side_thickness = float(self.config["tunnel_side_thickness"])
        self.tunnel_half_height = self.fluid_outlet_half_height
        self.tunnel_half_depth = self.fluid_outlet_half_depth
        self.tunnel_center = (
            self.fluid_outlet_center[0] + self.tunnel_half_length,
            self.fluid_outlet_center[1],
            self.fluid_outlet_center[2],
        )

        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self.tunnel_center[0], self.tunnel_center[1] - self.tunnel_half_height - self.tunnel_floor_half_thickness, self.tunnel_center[2]),
                wp.quat_identity(),
            ),
            hx=self.tunnel_half_length,
            hy=self.tunnel_floor_half_thickness,
            hz=self.tunnel_half_depth,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self.tunnel_center[0], self.tunnel_center[1] + self.tunnel_half_height + self.tunnel_top_half_thickness, self.tunnel_center[2]),
                wp.quat_identity(),
            ),
            hx=self.tunnel_half_length,
            hy=self.tunnel_top_half_thickness,
            hz=self.tunnel_half_depth + self.tunnel_side_thickness,
            cfg=wall_cfg,
        )
        for z_sign in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    (
                        self.tunnel_center[0],
                        self.tunnel_center[1],
                        self.tunnel_center[2] + z_sign * (self.tunnel_half_depth + self.tunnel_side_thickness),
                    ),
                    wp.quat_identity(),
                ),
                hx=self.tunnel_half_length,
                hy=self.tunnel_half_height + self.tunnel_top_half_thickness,
                hz=self.tunnel_side_thickness,
                cfg=wall_cfg,
            )

        self.fluid_gate_half_extents = (
            float(self.config["fluid_gate_half_thickness"]),
            self.fluid_outlet_half_height,
            self.fluid_outlet_half_depth,
        )
        gate_x = self.fluid_outlet_wall_center_x + self.fluid_gate_half_extents[0]
        self.fluid_gate_closed_center = (gate_x, self.fluid_outlet_center[1], self.fluid_outlet_center[2])
        self.fluid_gate_open_center = (
            self.fluid_gate_closed_center[0],
            float(self.config["fluid_gate_open_height"]),
            self.fluid_gate_closed_center[2],
        )
        self.fluid_gate_body = builder.add_body(
            xform=wp.transform(self.fluid_gate_closed_center, self.tunnel_rotation),
            is_kinematic=True,
            label="fsi_fluid_gate",
        )
        builder.add_shape_box(
            body=self.fluid_gate_body,
            hx=self.fluid_gate_half_extents[0],
            hy=self.fluid_gate_half_extents[1],
            hz=self.fluid_gate_half_extents[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
        )

    def _add_fluid_block(self, builder: newton.ModelBuilder) -> None:
        self.fluid_particle_start = builder.particle_count
        cell = float(self.config["fluid_cell"])
        dim_x = int(self.config["fluid_dim_x"])
        dim_y = int(self.config["fluid_dim_y"])
        dim_z = int(self.config["fluid_dim_z"])

        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=self.fluid_reservoir_center,
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell,
                cell_y=cell,
                cell_z=cell,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            mass=float(self.config["fluid_mass"]),
            jitter=0.0,
            radius_mean=float(self.config["fluid_radius"]),
            flags=int(newton.ParticleFlags.ACTIVE),
        )
        self.fluid_particle_count = builder.particle_count - self.fluid_particle_start

    def _set_particle_fixed(self, builder: newton.ModelBuilder, particle_idx: int) -> None:
        builder.particle_flags[particle_idx] = int(builder.particle_flags[particle_idx]) & ~int(newton.ParticleFlags.ACTIVE)
        builder.particle_mass[particle_idx] = 0.0

    def _add_corner_fixed_cloth(
        self,
        builder: newton.ModelBuilder,
        *,
        center: tuple[float, float, float],
        tilt_degrees: float,
    ) -> None:
        dim_x = int(self.config["cloth_dim_x"])
        dim_y = int(self.config["cloth_dim_y"])
        cell_x = float(self.config["cloth_cell_x"])
        cell_y = float(self.config["cloth_cell_y"])
        width = dim_x * cell_x
        height = dim_y * cell_y
        rotation = self._cloth_rotation(tilt_degrees)
        pos = tuple(np.array(center, dtype=np.float32) - _quat_rotate_np(rotation, (0.5 * width, 0.5 * height, 0.0)))

        start_particle = builder.particle_count
        if self.cloth_triangle_start is None:
            self.cloth_triangle_start = builder.tri_count

        builder.add_cloth_grid(
            pos=wp.vec3(*pos),
            rot=rotation,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=float(self.config["cloth_particle_mass"]),
            fix_left=False,
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=float(self.config["cloth_tri_ke"]),
            tri_ka=float(self.config["cloth_tri_ka"]),
            tri_kd=float(self.config["cloth_tri_kd"]),
            edge_ke=float(self.config["cloth_edge_ke"]),
            edge_kd=float(self.config["cloth_edge_kd"]),
            particle_radius=float(self.config["cloth_particle_radius"]),
        )

        stride = dim_x + 1
        corner_local_ids = (
            0,
            dim_x,
            dim_y * stride,
            dim_y * stride + dim_x,
        )
        for local_idx in corner_local_ids:
            self._set_particle_fixed(builder, start_particle + local_idx)

    def _add_cloth_cascade(self, builder: newton.ModelBuilder) -> None:
        self.cloth_particle_start = builder.particle_count
        self.cloth_triangle_start: int | None = None

        centers = tuple(tuple(float(v) for v in center) for center in self.config["cloth_centers"])
        tilts = tuple(float(v) for v in self.config["cloth_tilt_degrees"])
        for center, tilt in zip(centers, tilts, strict=True):
            self._add_corner_fixed_cloth(builder, center=center, tilt_degrees=tilt)

        self.cloth_particle_count = builder.particle_count - self.cloth_particle_start
        self.cloth_triangle_count = (
            0 if self.cloth_triangle_start is None else builder.tri_count - self.cloth_triangle_start
        )

    def _build_model(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.ke = float(self.config["shape_contact_ke"])
        builder.default_shape_cfg.kd = float(self.config["shape_contact_kd"])
        builder.default_shape_cfg.gap = float(self.config["shape_contact_gap"])

        self._add_tank(builder)
        self._add_fluid_reservoir(builder)
        self._add_fluid_tunnel(builder)
        self._add_fluid_block(builder)
        self._add_cloth_cascade(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

    def _build_cloth_self_contact_vertex_filtering_map(self) -> dict[int, list[int]] | None:
        if self.cloth_triangle_count <= 0 or self.fluid_particle_count <= 0 or self.cloth_triangle_start is None:
            return None

        cloth_triangle_ids = list(range(self.cloth_triangle_start, self.cloth_triangle_start + self.cloth_triangle_count))
        return dict.fromkeys(
            range(self.fluid_particle_start, self.fluid_particle_start + self.fluid_particle_count),
            cloth_triangle_ids,
        )

    def _build_solvers(self) -> None:
        cloth_self_contact_vertex_filtering_map = None
        if bool(self.config["cloth_self_contact_enabled"]):
            cloth_self_contact_vertex_filtering_map = self._build_cloth_self_contact_vertex_filtering_map()

        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE,
            include_static=True,
            include_dynamic=True,
            include_triangles=True,
            deformable_sample_thickness=float(self.config["fluid_radius"]) * 2.0,
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
                fsi_triangle_contact_enabled=True,
                fsi_triangle_contact_margin=0.0,
                fsi_triangle_contact_relaxation=float(self.config["triangle_contact_relaxation"]),
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
            particle_enable_self_contact=bool(self.config["cloth_self_contact_enabled"]),
            particle_self_contact_radius=float(self.config["cloth_self_contact_radius"]),
            particle_self_contact_margin=float(self.config["cloth_self_contact_margin"]),
            particle_external_vertex_contact_filtering_map=cloth_self_contact_vertex_filtering_map,
            fsi_boundary_model=self.boundary_model,
        )
        coupling_mode = getattr(self.args, "coupling_mode", "interlinked")
        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
            config=SolverFSI.Config(
                mode=(
                    SolverFSI.Config.CouplingMode.INTERLINKED
                    if coupling_mode == "interlinked"
                    else SolverFSI.Config.CouplingMode.LOOSE
                ),
                coupling_iterations=max(1, int(getattr(self.args, "coupling_iterations", 3))),
            ),
        )
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            self.args,
            soft_contact_margin=max(float(self.config["fluid_radius"]), float(self.config["cloth_particle_radius"])) * 2.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

    def _build_visualization_data(self) -> None:
        self.fluid_colors = wp.full(
            self.fluid_particle_count,
            value=wp.vec3(0.18, 0.60, 1.00),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.fluid_radii = wp.full(
            self.fluid_particle_count,
            value=float(self.config["fluid_radius"]) * float(self.config["water_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.shape_material = wp.array([wp.vec4(0.42, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.gate_colors = wp.array(
            [wp.vec3(0.82, 0.84, 0.88)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.cloth_mesh_indices = wp.array(self.model.tri_indices.numpy().reshape(-1), dtype=wp.int32, device=self.model.device)
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * 0.18,
            dtype=wp.float32,
            device=self.model.device,
        )
        sample_body = self.boundary_model.sample_body.numpy()
        boundary_colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        boundary_colors[sample_body < 0] = np.array([0.92, 0.94, 0.98], dtype=np.float32)
        boundary_colors[sample_body >= 0] = np.array([1.0, 0.45, 0.18], dtype=np.float32)
        self.boundary_colors = wp.array(boundary_colors, dtype=wp.vec3, device=self.model.device)

        self.tank_wire_starts, self.tank_wire_ends = build_box_wireframe(
            min_x=-float(self.config["tank_half_width"]),
            max_x=float(self.config["tank_half_width"]),
            min_y=self.floor_y,
            max_y=self.floor_y + 2.0 * float(self.config["tank_wall_height"]),
            min_z=-float(self.config["tank_half_depth"]),
            max_z=float(self.config["tank_half_depth"]),
            include_top=False,
            device=self.model.device,
        )
        self.fluid_reservoir_wire_starts, self.fluid_reservoir_wire_ends = build_box_wireframe(
            min_x=self.fluid_reservoir_center[0] - self.fluid_reservoir_half_width,
            max_x=self.fluid_reservoir_center[0] + self.fluid_reservoir_half_width,
            min_y=self.fluid_reservoir_center[1] - self.fluid_reservoir_half_height,
            max_y=self.fluid_reservoir_center[1] + self.fluid_reservoir_half_height,
            min_z=self.fluid_reservoir_center[2] - self.fluid_reservoir_half_depth,
            max_z=self.fluid_reservoir_center[2] + self.fluid_reservoir_half_depth,
            include_top=False,
            device=self.model.device,
        )
        self.tunnel_wire_starts, self.tunnel_wire_ends = build_box_wireframe(
            min_x=self.fluid_outlet_center[0],
            max_x=self.fluid_outlet_center[0] + 2.0 * self.tunnel_half_length,
            min_y=self.tunnel_center[1] - self.tunnel_half_height,
            max_y=self.tunnel_center[1] + self.tunnel_half_height,
            min_z=self.tunnel_center[2] - self.tunnel_half_depth,
            max_z=self.tunnel_center[2] + self.tunnel_half_depth,
            include_top=True,
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
        self.tank_region_min = np.array(
            [
                -float(self.config["tank_half_width"]) * 0.95,
                self.floor_y - 0.05,
                -float(self.config["tank_half_depth"]) * 0.95,
            ],
            dtype=np.float32,
        )
        self.tank_region_max = np.array(
            [
                float(self.config["tank_half_width"]) * 0.95,
                self.floor_y + min(0.24, 0.6 * float(self.config["tank_wall_height"])),
                float(self.config["tank_half_depth"]) * 0.95,
            ],
            dtype=np.float32,
        )

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

    def _set_body_pose(
        self,
        state: newton.State,
        body_idx: int,
        center: tuple[float, float, float],
        rotation: wp.quat,
        *,
        linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        quat = np.array(rotation, dtype=np.float32)

        body_q_np = state.body_q.numpy()
        body_q_np[body_idx, 0] = float(center[0])
        body_q_np[body_idx, 1] = float(center[1])
        body_q_np[body_idx, 2] = float(center[2])
        body_q_np[body_idx, 3:7] = quat
        state.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)

        body_qd_np = state.body_qd.numpy()
        body_qd_np[body_idx, 0:3] = np.array(linear_velocity, dtype=np.float32)
        body_qd_np[body_idx, 3:6] = np.array(angular_velocity, dtype=np.float32)
        state.body_qd = wp.array(body_qd_np, dtype=wp.spatial_vector, device=self.model.device)

    def _fluid_gate_is_open(self) -> bool:
        return self.frame_index >= int(self.config["fluid_release_step"])

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

    def _gate_alpha(self, release_step: int, duration_frames: int) -> float:
        if self.frame_index < release_step:
            return 0.0
        if duration_frames <= 1:
            return 1.0
        return float(self.frame_index - release_step + 1) / float(duration_frames)

    def _apply_gate_states(self, state: newton.State) -> None:
        fluid_alpha = self._gate_alpha(
            int(self.config["fluid_release_step"]),
            int(self.config["fluid_gate_open_duration_frames"]),
        )
        fluid_center = self._lerp_center(self.fluid_gate_closed_center, self.fluid_gate_open_center, fluid_alpha)
        self._set_body_pose(state, self.fluid_gate_body, fluid_center, self.tunnel_rotation)

    def reset(self):
        self.sim_time = 0.0
        self.frame_index = 0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self._apply_gate_states(self.state_0)
        self._apply_gate_states(self.state_1)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()

        particle_q = self.state_0.particle_q.numpy()
        self.initial_particle_q = particle_q.copy()
        self.max_cloth_dx = 0.0
        self.max_triangle_contact_pair_count = 0
        self.max_fsi_particle_inertia_offset = 0.0
        self.max_vertex_force_norm = 0.0
        self.max_density = 0.0
        self.max_particles_in_tank = 0
        self.states_remain_finite = True
        self.viewer._paused = True

    def capture(self):
        self.graph = None

    def _record_substep_diagnostics(self) -> None:
        triangle_pair_count = int(self.fluid_solver._triangle_contact_pair_count.numpy()[0])
        vertex_force = self.boundary_model.vertex_force.numpy()[self.cloth_particle_start :]
        fsi_inertia_offset = self.solid_solver.fsi_particle_inertia_offset.numpy()[self.cloth_particle_start :]
        density = self.state_0.ipbf.density.numpy()
        body_force_step_avg = self.boundary_model.body_force_step_avg.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        self.max_triangle_contact_pair_count = max(self.max_triangle_contact_pair_count, triangle_pair_count)
        self.max_fsi_particle_inertia_offset = max(
            self.max_fsi_particle_inertia_offset,
            float(np.linalg.norm(fsi_inertia_offset, axis=1).max()),
        )
        self.max_vertex_force_norm = max(self.max_vertex_force_norm, float(np.linalg.norm(vertex_force, axis=1).max()))
        self.max_density = max(self.max_density, float(np.max(density)))
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(vertex_force).all()
            and np.isfinite(fsi_inertia_offset).all()
            and np.isfinite(density).all()
            and np.isfinite(body_force_step_avg).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
            and np.isfinite(body_q).all()
            and np.isfinite(body_qd).all()
        )

        fluid_q = particle_q[self.fluid_particle_start : self.fluid_particle_start + self.fluid_particle_count]
        in_tank = np.logical_and.reduce(
            (
                fluid_q[:, 0] >= self.tank_region_min[0],
                fluid_q[:, 0] <= self.tank_region_max[0],
                fluid_q[:, 1] >= self.tank_region_min[1],
                fluid_q[:, 1] <= self.tank_region_max[1],
                fluid_q[:, 2] >= self.tank_region_min[2],
                fluid_q[:, 2] <= self.tank_region_max[2],
            )
        )
        self.max_particles_in_tank = max(self.max_particles_in_tank, int(np.count_nonzero(in_tank)))

    def _record_frame_diagnostics(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        cloth_q = particle_q[self.cloth_particle_start :]
        cloth_q0 = self.initial_particle_q[self.cloth_particle_start :]
        self.max_cloth_dx = max(self.max_cloth_dx, float(np.max(np.abs(cloth_q[:, 0] - cloth_q0[:, 0]))))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_gate_states(self.state_0)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            wp.launch(
                scale_velocities,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, float(self.config["velocity_damping"])],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, control=None, contacts=self.contacts, dt=self.sim_dt)
            self._record_substep_diagnostics()
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            reset_down = bool(self.viewer.is_key_down("r"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        self.simulate()
        self._record_frame_diagnostics()
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def test_final(self):
        assert self.states_remain_finite, "cascade scene produced non-finite diagnostics"
        assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
        expected_triangle_pairs = int(self.config["expected_min_triangle_pairs"])
        expected_cloth_dx = float(self.config["expected_min_cloth_dx"])
        assert (
            self.max_triangle_contact_pair_count >= expected_triangle_pairs
        ), f"cloth triangle-contact pairs stayed too low: {self.max_triangle_contact_pair_count}"
        if expected_triangle_pairs > 0:
            assert self.max_fsi_particle_inertia_offset > 0.0, "cloth triangle-contact deltas never reached VBD"
        if expected_cloth_dx > 0.0:
            assert self.max_cloth_dx > expected_cloth_dx, f"cloth cascade did not deform enough: {self.max_cloth_dx}"
        assert self.max_particles_in_tank >= int(
            self.config["expected_min_particles_in_tank"]
        ), f"water never reached the catch tank: {self.max_particles_in_tank}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if bool(self.config["show_domain_wire"]):
            self.viewer.log_lines(
                "/fsi/cascade_domain",
                self.domain_wire_starts,
                self.domain_wire_ends,
                colors=(0.22, 0.24, 0.28),
                width=0.004,
            )
        self.viewer.log_lines(
            "/fsi/cascade_tank_wire",
            self.tank_wire_starts,
            self.tank_wire_ends,
            colors=(0.88, 0.90, 0.95),
            width=0.010,
        )
        self.viewer.log_lines(
            "/fsi/cascade_fluid_reservoir_wire",
            self.fluid_reservoir_wire_starts,
            self.fluid_reservoir_wire_ends,
            colors=(0.78, 0.84, 0.92),
            width=0.008,
        )
        self.viewer.log_lines(
            "/fsi/cascade_tunnel_wire",
            self.tunnel_wire_starts,
            self.tunnel_wire_ends,
            colors=(0.80, 0.84, 0.90),
            width=0.008,
        )
        self.viewer.log_mesh(
            "/fsi/cascade_cloth_mesh",
            points=self.state_0.particle_q,
            indices=self.cloth_mesh_indices,
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
        )
        self.viewer.log_shapes(
            "/fsi/cascade_fluid_gate",
            newton.GeoType.BOX,
            self.fluid_gate_half_extents,
            self._body_xforms([self.fluid_gate_body]),
            self.gate_colors,
            self.shape_material,
        )
        self.viewer.log_points(
            "/fsi/cascade_water_points",
            points=wp.array(
                self.state_0.particle_q.numpy()[self.fluid_particle_start : self.fluid_particle_start + self.fluid_particle_count],
                dtype=wp.vec3,
                device=self.model.device,
            ),
            radii=self.fluid_radii,
            colors=self.fluid_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.log_points(
            "/fsi/cascade_boundary_samples",
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
