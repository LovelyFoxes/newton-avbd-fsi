###########################################################################
# Example FSI IPBF VBD Cloth Payload Rain
#
# A square catch tank sits under a four-corner-pinned VBD cloth sheet. Several
# AVBD rigid spheres and boxes with three densities rest on the cloth while a
# cascade-style elevated reservoir releases water through a bottom side outlet.
#
# Command: python -m newton.examples fsi_ipbf_vbd_cloth_payload_rain
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ipbf.common import build_box_wireframe, get_particle_grid_origin_from_center, scale_velocities
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD


def _horizontal_cloth_rotation() -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)


def _quat_rotate_np(quat: wp.quat, vec: tuple[float, float, float]) -> np.ndarray:
    rot = np.array(wp.quat_to_matrix(quat), dtype=np.float32).reshape(3, 3)
    return rot @ np.array(vec, dtype=np.float32)


class Example:
    """Rain-on-cloth scene with mixed-density rigid payloads."""

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            choices=["interlinked", "loose"],
            default="interlinked",
            help="FSI scheduler mode used for the cloth payload rain scene.",
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=4,
            help="Number of outer IPBF/AVBD feedback passes used per simulation substep.",
        )
        parser.add_argument("--release-step", type=int, default=None, help="Frame step at which the water gate opens.")
        parser.add_argument(
            "--gate-open-duration-frames",
            type=int,
            default=None,
            help="Number of frames used to retract the water gate after release.",
        )
        parser.add_argument(
            "--triangle-velocity-damping",
            type=float,
            default=None,
            help="Override the cloth tangential velocity damping used by triangle velocity projection.",
        )
        parser.add_argument(
            "--show-boundary-samples",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Render FSI boundary samples for debugging.",
        )
        parser.add_argument(
            "--disable-payload",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Disable the rigid spheres and boxes resting on the cloth.",
        )
        parser.add_argument(
            "--disable-tank",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Disable the lower catch tank under the cloth.",
        )
        parser.add_argument(
            "--disable-fluid",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Disable the fluid particles together with the rear reservoir and gate.",
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
                "tank_half_width": 0.30,
                "tank_half_depth": 0.30,
                "tank_wall_height": 0.40,
                "tank_wall_thickness": 0.025,
                "cloth_center": (0.0, 0.88, 0.0),
                "cloth_dim_x": 12,
                "cloth_dim_y": 12,
                "cloth_span_x": 0.96,
                "cloth_span_y": 0.96,
                "cloth_surface_density": 0.55,
                "cloth_particle_radius_ratio": 0.125,
                "cloth_tri_ke": 95.0,
                "cloth_tri_ka": 95.0,
                "cloth_tri_kd": 0.28,
                "cloth_edge_ke": 10.0,
                "cloth_edge_kd": 0.06,
                "cloth_self_contact_radius_ratio": 0.225,
                "cloth_self_contact_margin_ratio": 0.375,
                "payload_sphere_radius": 0.045,
                "payload_box_half_extents": (0.038, 0.038, 0.038),
                "payload_densities": (300.0, 750.0, 1250.0),
                "payload_mu": 0.35,
                "reservoir_center": (0.0, 1.26, -0.74),
                "reservoir_half_width": 0.13,
                "reservoir_half_height": 0.24,
                "reservoir_half_depth": 0.13,
                "reservoir_wall_thickness": 0.014,
                "outlet_half_width": 0.045,
                "outlet_half_height": 0.045,
                "tunnel_half_length": 0.34,
                "tunnel_floor_half_thickness": 0.018,
                "tunnel_top_half_thickness": 0.012,
                "tunnel_side_thickness": 0.012,
                "outlet_gate_half_thickness": 0.012,
                "gate_open_height": 1.70,
                "gate_open_duration_frames": 6,
                "release_step": 20,
                "fluid_cell": 0.024,
                "fluid_dim_x": 10,
                "fluid_dim_y": 16,
                "fluid_dim_z": 10,
                "fluid_mass": 0.013824,
                "fluid_radius": 0.009,
                "rest_density": 1000.0,
                "smoothing_radius": 0.070,
                "ipbf_iterations": 4,
                "vbd_iterations": 12,
                "sim_substeps": 5,
                "velocity_damping": 0.997,
                "viscosity_coefficient": 0.0020,
                "xsph_coefficient": 0.004,
                "boundary_velocity_damping": 1.0,
                "fsi_pressure_reaction_relaxation": 1.0,
                "static_boundary_weight": 0.5,
                "triangle_velocity_damping": 0.50,
                "triangle_contact_relaxation": 1.0,
                "boundary_spacing": 0.030,
                "water_render_radius_scale": 0.70,
                "boundary_render_radius_scale": 0.24,
                "shape_contact_ke": 3.0e4,
                "shape_contact_kd": 1.2e3,
                "shape_contact_gap": 0.006,
            }
        else:
            config = {
                "tank_half_width": 0.40,
                "tank_half_depth": 0.40,
                "tank_wall_height": 0.40,
                "tank_wall_thickness": 0.035,
                "cloth_center": (0.0, 0.96, 0.0),
                "cloth_dim_x": 32,
                "cloth_dim_y": 32,
                "cloth_span_x": 1.44,
                "cloth_span_y": 1.44,
                "cloth_surface_density": 0.30,
                "cloth_particle_radius_ratio": 0.125,
                "cloth_tri_ke": 480.0,
                "cloth_tri_ka": 480.0,
                "cloth_tri_kd": 0.48,
                "cloth_edge_ke": 32.0,
                "cloth_edge_kd": 0.12,
                "cloth_self_contact_radius_ratio": 0.225,
                "cloth_self_contact_margin_ratio": 0.375,
                "payload_sphere_radius": 0.056,
                "payload_box_half_extents": (0.040, 0.040, 0.040),
                "payload_densities": (250.0, 500.0, 750.0),
                "payload_mu": 0.35,
                "reservoir_center": (0.0, 1.60, -1.12),
                "reservoir_half_width": 0.24,
                "reservoir_half_height": 0.56,
                "reservoir_half_depth": 0.24,
                "reservoir_wall_thickness": 0.018,
                "outlet_half_width": 0.088,
                "outlet_half_height": 0.088,
                "tunnel_half_length": 0.20,
                "tunnel_floor_half_thickness": 0.020,
                "tunnel_top_half_thickness": 0.014,
                "tunnel_side_thickness": 0.014,
                "outlet_gate_half_thickness": 0.014,
                "gate_open_height": 2.40,
                "gate_open_duration_frames": 8,
                "release_step": 30,
                "fluid_cell": 0.018,
                "fluid_dim_x": 24,
                "fluid_dim_y": 54,
                "fluid_dim_z": 24,
                "fluid_mass": 0.0058,
                "fluid_radius": 0.0068,
                "rest_density": 1000.0,
                "smoothing_radius": 0.040,
                "ipbf_iterations": 4,
                "vbd_iterations": 12,
                "sim_substeps": 5,
                "velocity_damping": 0.999,
                "viscosity_coefficient": 0.0025,
                "xsph_coefficient": 0.006,
                "boundary_velocity_damping": 0.97,
                "fsi_pressure_reaction_relaxation": 1.5,
                "static_boundary_weight": 0.20,
                "triangle_velocity_damping": 0.50,
                "triangle_contact_relaxation": 1.0,
                "boundary_spacing": 0.020,
                "water_render_radius_scale": 0.70,
                "boundary_render_radius_scale": 0.24,
                "shape_contact_ke": 4.0e4,
                "shape_contact_kd": 1.5e3,
                "shape_contact_gap": 0.006,
            }

        release_step = getattr(self.args, "release_step", None)
        if release_step is not None:
            config["release_step"] = max(0, int(release_step))

        gate_open_duration_frames = getattr(self.args, "gate_open_duration_frames", None)
        if gate_open_duration_frames is not None:
            config["gate_open_duration_frames"] = max(1, int(gate_open_duration_frames))

        triangle_velocity_damping = getattr(self.args, "triangle_velocity_damping", None)
        if triangle_velocity_damping is not None:
            config["triangle_velocity_damping"] = float(triangle_velocity_damping)

        return self._finalize_cloth_config(config)

    @staticmethod
    def _finalize_cloth_config(config: dict[str, object]) -> dict[str, object]:
        dim_x = max(1, int(config["cloth_dim_x"]))
        dim_y = max(1, int(config["cloth_dim_y"]))
        if "cloth_span_x" in config:
            span_x = float(config["cloth_span_x"])
        else:
            span_x = float(config["cloth_cell_x"]) * dim_x

        if "cloth_span_y" in config:
            span_y = float(config["cloth_span_y"])
        else:
            span_y = float(config["cloth_cell_y"]) * dim_y

        cell_x = span_x / dim_x
        cell_y = span_y / dim_y
        config["cloth_cell_x"] = cell_x
        config["cloth_cell_y"] = cell_y

        particle_count = (dim_x + 1) * (dim_y + 1)
        surface_density = config.get("cloth_surface_density", None)
        if surface_density is not None:
            total_mass = float(surface_density) * span_x * span_y
            config["cloth_particle_mass"] = total_mass / particle_count

        spacing = min(cell_x, cell_y)
        particle_radius_ratio = config.get("cloth_particle_radius_ratio", None)
        if particle_radius_ratio is not None:
            config["cloth_particle_radius"] = float(particle_radius_ratio) * spacing

        self_contact_radius_ratio = config.get("cloth_self_contact_radius_ratio", None)
        if self_contact_radius_ratio is not None:
            config["cloth_self_contact_radius"] = float(self_contact_radius_ratio) * spacing

        self_contact_margin_ratio = config.get("cloth_self_contact_margin_ratio", None)
        if self_contact_margin_ratio is not None:
            config["cloth_self_contact_margin"] = float(self_contact_margin_ratio) * spacing

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
        self.disable_payload = bool(getattr(self.args, "disable_payload", False))
        self.disable_tank = bool(getattr(self.args, "disable_tank", False))
        self.disable_fluid = bool(getattr(self.args, "disable_fluid", False))
        self.config = self._get_scene_config()
        self.sim_substeps = int(self.config["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.floor_y = 0.0
        self.enable_runtime_diagnostics = bool(getattr(self.args, "enable_runtime_diagnostics", False)) or bool(
            getattr(self.args, "test", False)
        )

        self._build_model()
        self._build_solvers()
        self._build_visualization_data()

        self.viewer.set_model(self.model)
        self.viewer.show_visual = False
        self.viewer.show_triangles = True
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.15, 1.50, 1.55), pitch=-20.0, yaw=-138.0)

        self.reset()
        self.capture()

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text("Cloth payload rain scene")
        ui.text(f"Mode: {getattr(self.args, 'coupling_mode', 'interlinked')}")
        ui.text(f"Frame: {self.frame_index}")
        ui.text(f"Water gate release: {int(self.config['release_step'])}")
        ui.text(f"Water gate open: {'yes' if self._gate_is_open() else 'no'}")
        ui.text(f"Disable fluid: {'yes' if self.disable_fluid else 'no'}")
        ui.text(f"Disable payload: {'yes' if self.disable_payload else 'no'}")
        ui.text(f"Disable tank: {'yes' if self.disable_tank else 'no'}")
        ui.text(f"Rigid payloads: {len(self.sphere_bodies) + len(self.box_bodies)}")
        ui.text(f"Payload densities: {tuple(float(v) for v in self.config['payload_densities'])}")
        ui.text(f"Cloth particles: {self.cloth_particle_count}")
        ui.text(f"Cloth total mass: {self.cloth_particle_count * float(self.config['cloth_particle_mass']):.3f} kg")

    def _add_catch_tank(self, builder: newton.ModelBuilder) -> None:
        hx = float(self.config["tank_half_width"])
        hz = float(self.config["tank_half_depth"])
        hy = float(self.config["tank_wall_height"])
        wt = float(self.config["tank_wall_thickness"])
        self.tank_top_y = self.floor_y + 2.0 * hy

        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
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

    def _add_side_outlet_reservoir(self, builder: newton.ModelBuilder) -> None:
        cx, cy, cz = tuple(float(v) for v in self.config["reservoir_center"])
        hx = float(self.config["reservoir_half_width"])
        hy = float(self.config["reservoir_half_height"])
        hz = float(self.config["reservoir_half_depth"])
        wt = float(self.config["reservoir_wall_thickness"])
        outlet_hx = float(self.config["outlet_half_width"])
        outlet_hy = float(self.config["outlet_half_height"])

        self.reservoir_center = (cx, cy, cz)
        self.reservoir_half_extents = (hx, hy, hz)
        self.reservoir_bottom_y = cy - hy
        self.outlet_center = (cx, self.reservoir_bottom_y + outlet_hy, cz + hz)
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, self.reservoir_bottom_y - wt, cz), wp.quat_identity()),
            hx=hx + wt,
            hy=wt,
            hz=hz + wt,
            cfg=wall_cfg,
        )
        for x_sign in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform((cx + x_sign * (hx + wt), cy, cz), wp.quat_identity()),
                hx=wt,
                hy=hy + wt,
                hz=hz + wt,
                cfg=wall_cfg,
            )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((cx, cy, cz - hz - wt), wp.quat_identity()),
            hx=hx + wt,
            hy=hy + wt,
            hz=wt,
            cfg=wall_cfg,
        )

        outlet_y = self.outlet_center[1]
        front_wall_z = cz + hz + wt
        upper_bottom = outlet_y + outlet_hy
        if upper_bottom < cy + hy:
            upper_half_height = 0.5 * (cy + hy - upper_bottom)
            builder.add_shape_box(
                body=-1,
                xform=wp.transform((cx, upper_bottom + upper_half_height, front_wall_z), wp.quat_identity()),
                hx=hx + wt,
                hy=upper_half_height,
                hz=wt,
                cfg=wall_cfg,
            )

        side_half_width = 0.5 * (hx - outlet_hx)
        if side_half_width > 0.0:
            for x_sign in (-1.0, 1.0):
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(
                        (cx + x_sign * (outlet_hx + side_half_width), outlet_y, front_wall_z),
                        wp.quat_identity(),
                    ),
                    hx=side_half_width,
                    hy=outlet_hy,
                    hz=wt,
                    cfg=wall_cfg,
                )

        tunnel_half_length = float(self.config["tunnel_half_length"])
        tunnel_floor_half_thickness = float(self.config["tunnel_floor_half_thickness"])
        tunnel_top_half_thickness = float(self.config["tunnel_top_half_thickness"])
        tunnel_side_thickness = float(self.config["tunnel_side_thickness"])
        self.tunnel_half_width = outlet_hx
        self.tunnel_half_height = outlet_hy
        self.tunnel_half_length = tunnel_half_length
        self.tunnel_center = (cx, outlet_y, cz + hz + tunnel_half_length)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self.tunnel_center[0], outlet_y - outlet_hy - tunnel_floor_half_thickness, self.tunnel_center[2]),
                wp.quat_identity(),
            ),
            hx=outlet_hx,
            hy=tunnel_floor_half_thickness,
            hz=tunnel_half_length,
            cfg=wall_cfg,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self.tunnel_center[0], outlet_y + outlet_hy + tunnel_top_half_thickness, self.tunnel_center[2]),
                wp.quat_identity(),
            ),
            hx=outlet_hx + tunnel_side_thickness,
            hy=tunnel_top_half_thickness,
            hz=tunnel_half_length,
            cfg=wall_cfg,
        )
        for x_sign in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    (cx + x_sign * (outlet_hx + tunnel_side_thickness), outlet_y, self.tunnel_center[2]),
                    wp.quat_identity(),
                ),
                hx=tunnel_side_thickness,
                hy=outlet_hy + tunnel_top_half_thickness,
                hz=tunnel_half_length,
                cfg=wall_cfg,
            )

        gate_hz = float(self.config["outlet_gate_half_thickness"])
        self.gate_half_extents = (outlet_hx, outlet_hy, gate_hz)
        self.gate_closed_center = (cx, outlet_y, cz + hz + gate_hz)
        self.gate_open_center = (cx, float(self.config["gate_open_height"]), self.gate_closed_center[2])
        self.gate_body = builder.add_body(
            xform=wp.transform(self.gate_closed_center, wp.quat_identity()),
            is_kinematic=True,
            label="fsi_cloth_payload_rain_gate",
        )
        builder.add_shape_box(
            body=self.gate_body,
            hx=self.gate_half_extents[0],
            hy=self.gate_half_extents[1],
            hz=self.gate_half_extents[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0),
        )

    def _add_rigid_payloads(self, builder: newton.ModelBuilder) -> None:
        cloth_y = float(self.config["cloth_center"][1])
        sphere_radius = float(self.config["payload_sphere_radius"])
        box_hx, box_hy, box_hz = tuple(float(v) for v in self.config["payload_box_half_extents"])
        densities = tuple(float(v) for v in self.config["payload_densities"])
        mu = float(self.config["payload_mu"])

        sphere_positions = (
            (-0.17, cloth_y + sphere_radius + 0.012, -0.04),
            (0.00, cloth_y + sphere_radius + 0.018, 0.08),
            (0.17, cloth_y + sphere_radius + 0.012, -0.04),
        )
        box_positions = (
            (-0.12, cloth_y + box_hy + 0.016, 0.18),
            (0.12, cloth_y + box_hy + 0.016, 0.18),
            (0.00, cloth_y + box_hy + 0.018, -0.16),
        )

        self.sphere_bodies: list[int] = []
        self.box_bodies: list[int] = []
        for index, (position, density) in enumerate(zip(sphere_positions, densities, strict=True)):
            body = builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=f"fsi_payload_sphere_{index}")
            builder.add_shape_sphere(
                body=body,
                radius=sphere_radius,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=mu),
            )
            self.sphere_bodies.append(body)

        for index, (position, density) in enumerate(zip(box_positions, densities, strict=True)):
            body = builder.add_body(
                xform=wp.transform(
                    position,
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.35 * float(index + 1)),
                ),
                label=f"fsi_payload_box_{index}",
            )
            builder.add_shape_box(
                body=body,
                hx=box_hx,
                hy=box_hy,
                hz=box_hz,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=mu),
            )
            self.box_bodies.append(body)

    def _add_fluid_block(self, builder: newton.ModelBuilder) -> None:
        self.fluid_particle_start = builder.particle_count
        cell = float(self.config["fluid_cell"])
        dim_x = int(self.config["fluid_dim_x"])
        dim_y = int(self.config["fluid_dim_y"])
        dim_z = int(self.config["fluid_dim_z"])

        builder.add_particle_grid(
            pos=get_particle_grid_origin_from_center(
                center=self.reservoir_center,
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

    def _add_cloth_sheet(self, builder: newton.ModelBuilder) -> None:
        self.cloth_particle_start = builder.particle_count
        self.cloth_triangle_start = builder.tri_count
        dim_x = int(self.config["cloth_dim_x"])
        dim_y = int(self.config["cloth_dim_y"])
        cell_x = float(self.config["cloth_cell_x"])
        cell_y = float(self.config["cloth_cell_y"])
        width = dim_x * cell_x
        height = dim_y * cell_y
        rotation = _horizontal_cloth_rotation()
        center = tuple(float(v) for v in self.config["cloth_center"])
        pos = tuple(np.array(center, dtype=np.float32) - _quat_rotate_np(rotation, (0.5 * width, 0.5 * height, 0.0)))

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
        for local_idx in (0, dim_x, dim_y * stride, dim_y * stride + dim_x):
            self._set_particle_fixed(builder, self.cloth_particle_start + local_idx)

        self.cloth_particle_count = builder.particle_count - self.cloth_particle_start
        self.cloth_triangle_count = builder.tri_count - self.cloth_triangle_start

    def _build_model(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.ke = float(self.config["shape_contact_ke"])
        builder.default_shape_cfg.kd = float(self.config["shape_contact_kd"])
        builder.default_shape_cfg.gap = float(self.config["shape_contact_gap"])

        self.sphere_bodies = []
        self.box_bodies = []
        self.gate_body = None
        self.fluid_particle_start = builder.particle_count
        self.fluid_particle_count = 0

        if not self.disable_tank:
            self._add_catch_tank(builder)
        else:
            self.tank_top_y = self.floor_y
        if not self.disable_fluid:
            self._add_side_outlet_reservoir(builder)
            self._add_fluid_block(builder)
        if not self.disable_payload:
            self._add_rigid_payloads(builder)
        self._add_cloth_sheet(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

    def _build_cloth_self_contact_vertex_filtering_map(self) -> dict[int, list[int]] | None:
        if self.cloth_triangle_count <= 0 or self.cloth_triangle_start is None:
            return None
        cloth_triangle_ids = list(range(self.cloth_triangle_start, self.cloth_triangle_start + self.cloth_triangle_count))
        return dict.fromkeys(range(self.cloth_particle_start, self.model.particle_count), cloth_triangle_ids)

    def _build_solvers(self) -> None:
        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=float(self.config["boundary_spacing"]),
            support_radius=float(self.config["smoothing_radius"]),
            include_static=True,
            include_dynamic=True,
            include_triangles=True,
            deformable_sample_thickness=float(self.config["fluid_radius"]) * 2.0,
            device=self.model.device,
        )
        self.boundary_model.set_triangle_boundary_samples_active(False)
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
                fsi_triangle_pressure_reaction_enabled=False,
                fsi_triangle_velocity_projection_enabled=True,
                fsi_triangle_velocity_damping=float(self.config["triangle_velocity_damping"]),
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
            particle_enable_self_contact=False,
            particle_self_contact_radius=float(self.config["cloth_self_contact_radius"]),
            particle_self_contact_margin=float(self.config["cloth_self_contact_margin"]),
            particle_external_vertex_contact_filtering_map=self._build_cloth_self_contact_vertex_filtering_map(),
            rigid_body_contact_buffer_size=96,
            rigid_body_particle_contact_buffer_size=384,
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

    def _body_xforms(self, body_indices: list[int]) -> wp.array(dtype=wp.transform):
        body_q = self.state_0.body_q.numpy()[body_indices]
        return wp.array(body_q, dtype=wp.transform, device=self.model.device)

    def _build_boundary_colors(self) -> wp.array(dtype=wp.vec3):
        sample_body = self.boundary_model.sample_body.numpy()
        colors = np.zeros((self.boundary_model.sample_count, 3), dtype=np.float32)
        colors[sample_body < 0] = np.array([0.88, 0.90, 0.95], dtype=np.float32)
        palette = (
            np.array([1.0, 0.72, 0.24], dtype=np.float32),
            np.array([0.30, 0.86, 0.34], dtype=np.float32),
            np.array([0.18, 0.80, 1.00], dtype=np.float32),
        )
        for index, body in enumerate([*self.sphere_bodies, *self.box_bodies]):
            colors[sample_body == body] = palette[index % len(palette)]
        return wp.array(colors, dtype=wp.vec3, device=self.model.device)

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
        self.boundary_colors = self._build_boundary_colors()
        self.boundary_radii = wp.full(
            self.boundary_model.sample_count,
            value=float(self.config["boundary_spacing"]) * float(self.config["boundary_render_radius_scale"]),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.sphere_colors = wp.array(
            [wp.vec3(1.0, 0.72, 0.24), wp.vec3(0.30, 0.86, 0.34), wp.vec3(0.18, 0.80, 1.00)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.box_colors = wp.array(
            [wp.vec3(1.0, 0.46, 0.20), wp.vec3(0.54, 0.88, 0.28), wp.vec3(0.16, 0.58, 0.88)],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.shape_material = wp.array([wp.vec4(0.42, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.cloth_mesh_indices = wp.array(self.model.tri_indices.numpy().reshape(-1), dtype=wp.int32, device=self.model.device)
        self.tank_wire_starts = None
        self.tank_wire_ends = None
        if not self.disable_tank:
            hx = float(self.config["tank_half_width"])
            hz = float(self.config["tank_half_depth"])
            self.tank_wire_starts, self.tank_wire_ends = build_box_wireframe(
                min_x=-hx,
                max_x=hx,
                min_y=self.floor_y,
                max_y=self.tank_top_y,
                min_z=-hz,
                max_z=hz,
                include_top=False,
                device=self.model.device,
            )

        self.reservoir_wire_starts = None
        self.reservoir_wire_ends = None
        self.tunnel_wire_starts = None
        self.tunnel_wire_ends = None
        if not self.disable_fluid:
            rx, ry, rz = self.reservoir_center
            rhx, rhy, rhz = self.reservoir_half_extents
            self.reservoir_wire_starts, self.reservoir_wire_ends = build_box_wireframe(
                min_x=rx - rhx,
                max_x=rx + rhx,
                min_y=ry - rhy,
                max_y=ry + rhy,
                min_z=rz - rhz,
                max_z=rz + rhz,
                include_top=False,
                device=self.model.device,
            )
            self.tunnel_wire_starts, self.tunnel_wire_ends = build_box_wireframe(
                min_x=self.tunnel_center[0] - self.tunnel_half_width,
                max_x=self.tunnel_center[0] + self.tunnel_half_width,
                min_y=self.tunnel_center[1] - self.tunnel_half_height,
                max_y=self.tunnel_center[1] + self.tunnel_half_height,
                min_z=self.tunnel_center[2] - self.tunnel_half_length,
                max_z=self.tunnel_center[2] + self.tunnel_half_length,
                include_top=True,
                device=self.model.device,
            )

    def _gate_is_open(self) -> bool:
        if self.disable_fluid or self.gate_body is None:
            return False
        return self.frame_index >= int(self.config["release_step"])

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
        duration_frames = int(self.config["gate_open_duration_frames"])
        if self.frame_index < release_step:
            return 0.0
        if duration_frames <= 1:
            return 1.0
        return float(self.frame_index - release_step + 1) / float(duration_frames)

    def _set_body_pose(
        self,
        state: newton.State,
        body_idx: int,
        center: tuple[float, float, float],
        rotation: wp.quat,
    ) -> None:
        body_q_np = state.body_q.numpy()
        body_q_np[body_idx, 0:3] = np.array(center, dtype=np.float32)
        body_q_np[body_idx, 3:7] = np.array(rotation, dtype=np.float32)
        state.body_q = wp.array(body_q_np, dtype=wp.transform, device=self.model.device)

        body_qd_np = state.body_qd.numpy()
        body_qd_np[body_idx, 0:6] = np.zeros(6, dtype=np.float32)
        state.body_qd = wp.array(body_qd_np, dtype=wp.spatial_vector, device=self.model.device)

    def _apply_gate_state(self, state: newton.State) -> None:
        if self.disable_fluid or self.gate_body is None:
            return
        gate_center = self._lerp_center(self.gate_closed_center, self.gate_open_center, self._gate_alpha())
        self._set_body_pose(state, self.gate_body, gate_center, wp.quat_identity())

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

        self.max_density = 0.0
        self.max_particle_speed = 0.0
        self.max_body_speed = 0.0
        self.max_triangle_contact_pair_count = 0
        self.states_remain_finite = True
        self.viewer._paused = True

    def capture(self):
        self.graph = None

    def _record_diagnostics(self) -> None:
        density = self.state_0.ipbf.density.numpy()
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        body_q = None if self.state_0.body_q is None else self.state_0.body_q.numpy()
        body_qd = None if self.state_0.body_qd is None else self.state_0.body_qd.numpy()
        triangle_pair_count = int(self.fluid_solver._triangle_contact_pair_count.numpy()[0])

        if density.size:
            self.max_density = max(self.max_density, float(np.max(density)))
        if particle_qd.size:
            self.max_particle_speed = max(self.max_particle_speed, float(np.linalg.norm(particle_qd, axis=1).max()))
        if body_qd is not None and body_qd.size:
            self.max_body_speed = max(self.max_body_speed, float(np.linalg.norm(body_qd[:, :3], axis=1).max()))
        self.max_triangle_contact_pair_count = max(self.max_triangle_contact_pair_count, triangle_pair_count)
        self.states_remain_finite = self.states_remain_finite and bool(
            np.isfinite(density).all()
            and np.isfinite(particle_q).all()
            and np.isfinite(particle_qd).all()
            and (body_q is None or np.isfinite(body_q).all())
            and (body_qd is None or np.isfinite(body_qd).all())
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_gate_state(self.state_0)
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
            if self.enable_runtime_diagnostics:
                self._record_diagnostics()
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            reset_down = bool(self.viewer.is_key_down("r"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        self.simulate()
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def test_final(self):
        assert self.states_remain_finite, "simulation produced non-finite particle/body state"
        if not self.disable_fluid:
            assert self.max_density > 0.0, "IPBF density diagnostics were not updated"
            assert self.max_density < 2.8 * float(self.config["rest_density"]), (
                f"peak density is too large: {self.max_density:.3f}"
            )
        if self.model.particle_count > 0:
            assert self.max_particle_speed < 30.0, f"particle speed blew up: vmax={self.max_particle_speed:.3f}"
        assert self.max_body_speed < 25.0, f"rigid payload speed blew up: vmax={self.max_body_speed:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.tank_wire_starts is not None and self.tank_wire_ends is not None:
            self.viewer.log_lines(
                "/fsi/cloth_payload_rain_tank_wire",
                self.tank_wire_starts,
                self.tank_wire_ends,
                colors=(0.88, 0.90, 0.95),
                width=0.010,
            )
        if self.reservoir_wire_starts is not None and self.reservoir_wire_ends is not None:
            self.viewer.log_lines(
                "/fsi/cloth_payload_rain_reservoir_wire",
                self.reservoir_wire_starts,
                self.reservoir_wire_ends,
                colors=(0.78, 0.84, 0.92),
                width=0.008,
            )
        if self.tunnel_wire_starts is not None and self.tunnel_wire_ends is not None:
            self.viewer.log_lines(
                "/fsi/cloth_payload_rain_tunnel_wire",
                self.tunnel_wire_starts,
                self.tunnel_wire_ends,
                colors=(0.80, 0.84, 0.90),
                width=0.008,
            )
        self.viewer.log_mesh(
            "/fsi/cloth_payload_rain_cloth",
            points=self.state_0.particle_q,
            indices=self.cloth_mesh_indices,
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
        )
        if self.gate_body is not None:
            self.viewer.log_shapes(
                "/fsi/cloth_payload_rain_gate",
                newton.GeoType.BOX,
                self.gate_half_extents,
                self._body_xforms([self.gate_body]),
                wp.array([wp.vec3(0.82, 0.84, 0.88)], dtype=wp.vec3, device=self.model.device),
                self.shape_material,
            )
        if self.sphere_bodies:
            self.viewer.log_shapes(
                "/fsi/cloth_payload_rain_spheres",
                newton.GeoType.SPHERE,
                float(self.config["payload_sphere_radius"]),
                self._body_xforms(self.sphere_bodies),
                self.sphere_colors,
                self.shape_material,
            )
        if self.box_bodies:
            self.viewer.log_shapes(
                "/fsi/cloth_payload_rain_boxes",
                newton.GeoType.BOX,
                tuple(float(v) for v in self.config["payload_box_half_extents"]),
                self._body_xforms(self.box_bodies),
                self.box_colors,
                self.shape_material,
            )
        if self.fluid_particle_count > 0:
            self.viewer.log_points(
                "/fsi/cloth_payload_rain_water",
                points=wp.array(
                    self.state_0.particle_q.numpy()[
                        self.fluid_particle_start : self.fluid_particle_start + self.fluid_particle_count
                    ],
                    dtype=wp.vec3,
                    device=self.model.device,
                ),
                radii=self.fluid_radii,
                colors=self.fluid_colors,
                hidden=not self.viewer.show_particles,
            )
        self.viewer.log_points(
            "/fsi/cloth_payload_rain_boundary_samples",
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
