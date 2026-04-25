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

"""Export headless Newton FSI simulation caches for offline rendering.

This tool intentionally does not import Newton examples or viewer code. It
builds the target scene directly, advances the public Newton solvers headlessly,
and writes a renderer-friendly cache plus rigorous simulation timing metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import FSIBoundaryModel, SolverFSI, SolverIPBF, SolverVBD


@wp.kernel
def _scale_velocities_kernel(particle_qd: wp.array(dtype=wp.vec3), scale: float):
    tid = wp.tid()
    particle_qd[tid] = scale * particle_qd[tid]


@dataclass(frozen=True)
class SceneConfig:
    container_half_width: float = 0.52
    container_half_depth: float = 0.39
    wall_half_height: float = 0.92
    wall_thickness: float = 0.05
    floor_y: float = 0.0
    pool_dim_x: int = 64
    pool_dim_y: int = 30
    pool_dim_z: int = 48
    cell: float = 0.014
    particle_mass: float = 0.002744
    particle_radius: float = 0.006
    smoothing_radius: float = 0.025
    pool_bottom_clearance: float = 0.03
    sphere_radius: float = 0.07
    sphere_bottom_gap: float = 0.035
    sphere_densities: tuple[float, float, float] = (250.0, 700.0, 1250.0)
    rest_density: float = 1000.0
    fps: int = 60
    sim_substeps: int = 6
    ipbf_iterations: int = 2
    rigid_iterations: int = 2
    coupling_iterations: int = 3
    velocity_damping: float = 0.999
    boundary_spacing: float = 0.014
    coupling_mode: str = "interlinked"
    hydrostatic_volume_mode: str = "dynamic-shape-surface-quadrature"
    include_static_boundary_samples: bool = False
    fsi_projection_reaction_relaxation: float = 0.0
    fsi_velocity_projection_reaction_relaxation: float = 0.0
    fsi_pressure_reaction_relaxation: float = 2.0
    fsi_static_boundary_weight: float = 1.0

    @property
    def frame_dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / float(self.sim_substeps)

    @property
    def particle_count(self) -> int:
        return self.pool_dim_x * self.pool_dim_y * self.pool_dim_z


@dataclass
class ExportStats:
    sim_ms: list[float]
    export_ms: list[float]
    warmup_ms: list[float]

    def timing_summary(self, sim_substeps: int, warmup_frames: int) -> dict[str, Any]:
        sim = np.asarray(self.sim_ms, dtype=np.float64)
        export = np.asarray(self.export_ms, dtype=np.float64)
        avg_sim_ms = float(np.mean(sim)) if sim.size else 0.0
        return {
            "avg_sim_ms_per_frame": avg_sim_ms,
            "min_sim_ms_per_frame": float(np.min(sim)) if sim.size else 0.0,
            "max_sim_ms_per_frame": float(np.max(sim)) if sim.size else 0.0,
            "avg_sim_ms_per_substep": avg_sim_ms / float(max(1, sim_substeps)),
            "sim_frame_ms": [float(value) for value in sim],
            "avg_export_ms_per_recorded_frame": float(np.mean(export)) if export.size else 0.0,
            "min_export_ms_per_recorded_frame": float(np.min(export)) if export.size else 0.0,
            "max_export_ms_per_recorded_frame": float(np.max(export)) if export.size else 0.0,
            "export_frame_ms": [float(value) for value in export],
            "timing_scope": (
                "Simulation timing measures SolverFSI advancement only, with device synchronization before and after "
                "each Newton engine frame. Export timing is measured separately and includes array copies plus NPZ I/O."
            ),
            "timing_uses_device_synchronize": True,
            "performance_excludes_viewer": True,
            "performance_excludes_blender_rendering": True,
            "performance_excludes_one_time_warp_module_load": warmup_frames > 0,
            "one_time_warp_module_load_note": (
                "Use at least one warmup frame when reporting performance so Warp module load/first-launch overhead is "
                "not included in avg_sim_ms_per_frame."
            ),
        }


class ThreeSphereBuoyScene:
    """Headless three-sphere-buoy FSI scene built without viewer/example code."""

    def __init__(self, config: SceneConfig, device: wp.context.Device):
        self.config = config
        self.device = device
        self.sim_time = 0.0
        self.sim_frame_index = 0
        self.floor_y = config.floor_y
        self.top_y = config.floor_y + 2.0 * config.wall_half_height

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.0

        self._add_container(builder)
        self.buoy_bodies = self._add_sphere_buoys(builder)
        self._add_pool(builder)
        builder.color()

        self.model = builder.finalize(device=device)
        self.model.set_gravity((0.0, -9.81, 0.0))
        self.boundary_model = FSIBoundaryModel(
            self.model,
            spacing=config.boundary_spacing,
            support_radius=config.smoothing_radius,
            hydrostatic_volume_mode=_parse_hydrostatic_volume_mode(config.hydrostatic_volume_mode),
            include_static=config.include_static_boundary_samples,
            include_dynamic=True,
            device=self.model.device,
        )
        self.fluid_solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=config.rest_density,
                smoothing_radius=config.smoothing_radius,
                compliance=1.0e-5,
                iterations=config.ipbf_iterations,
                relaxation=0.5,
                viscosity_coefficient=0.0025,
                xsph_coefficient=0.005,
                fsi_projection_reaction_relaxation=config.fsi_projection_reaction_relaxation,
                fsi_velocity_projection_reaction_relaxation=config.fsi_velocity_projection_reaction_relaxation,
                fsi_pressure_reaction_relaxation=config.fsi_pressure_reaction_relaxation,
                fsi_static_boundary_weight=config.fsi_static_boundary_weight,
            ),
            boundary_model=self.boundary_model,
        )
        self.solid_solver = SolverVBD(
            self.model,
            iterations=config.rigid_iterations,
            integrate_particles=False,
            fsi_boundary_model=self.boundary_model,
        )
        self.solver = SolverFSI(
            self.model,
            fluid_solver=self.fluid_solver,
            solid_solver=self.solid_solver,
            boundary_model=self.boundary_model,
            config=SolverFSI.Config(
                mode=_parse_coupling_mode(config.coupling_mode),
                coupling_iterations=max(1, config.coupling_iterations),
            ),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=config.particle_radius * 2.0,
        )
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.reset()

    def reset(self) -> None:
        self.sim_time = 0.0
        self.sim_frame_index = 0
        self.fluid_solver.reset(self.state_0)
        self.fluid_solver.reset(self.state_1)
        self.solid_solver.reset(self.state_0)
        self.solid_solver.reset(self.state_1)
        self.contacts.clear()
        self.boundary_model.clear_forces()
        self.boundary_model.update_world_kinematics(self.state_0)
        self.boundary_model.build_grid()

    def simulate_frame(self) -> None:
        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _scale_velocities_kernel,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.config.velocity_damping],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, control=None, contacts=self.contacts, dt=self.config.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_frame_index += 1
        self.sim_time += self.config.frame_dt

    def make_frame_payload(
        self,
        *,
        record_index: int,
        save_velocities: bool,
        save_density: bool,
        save_boundary_samples: bool,
    ) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "record_index": np.asarray(record_index, dtype=np.int32),
            "sim_frame_index": np.asarray(self.sim_frame_index, dtype=np.int32),
            "sim_time": np.asarray(self.sim_time, dtype=np.float64),
            "fluid_positions": self.state_0.particle_q.numpy().astype(np.float32, copy=False),
            "fluid_radii": self.model.particle_radius.numpy().astype(np.float32, copy=False),
            "body_q": self.state_0.body_q.numpy().astype(np.float32, copy=False),
            "body_qd": self.state_0.body_qd.numpy().astype(np.float32, copy=False),
            "body_force_step_avg": self.boundary_model.body_force_step_avg.numpy().astype(np.float32, copy=False),
            "body_torque_step_avg": self.boundary_model.body_torque_step_avg.numpy().astype(np.float32, copy=False),
        }
        if save_velocities:
            payload["fluid_velocities"] = self.state_0.particle_qd.numpy().astype(np.float32, copy=False)
        if save_density and hasattr(self.state_0, "ipbf"):
            payload["fluid_density"] = self.state_0.ipbf.density.numpy().astype(np.float32, copy=False)
        if save_boundary_samples:
            payload["boundary_sample_positions"] = self.boundary_model.sample_x_world.numpy().astype(
                np.float32, copy=False
            )
            payload["boundary_sample_velocities"] = self.boundary_model.sample_v_world.numpy().astype(
                np.float32, copy=False
            )
            payload["boundary_sample_body"] = self.boundary_model.sample_body.numpy().astype(np.int32, copy=False)
            payload["boundary_sample_volume"] = self.boundary_model.sample_volume.numpy().astype(np.float32, copy=False)
            payload["boundary_sample_volume_hydrostatic"] = (
                self.boundary_model.sample_volume_hydrostatic.numpy().astype(np.float32, copy=False)
            )
        return payload

    def metadata_base(self) -> dict[str, Any]:
        body_masses = self.model.body_mass.numpy().astype(np.float64).tolist()
        buoy_body_indices = [int(body) for body in self.buoy_bodies]
        return {
            "scene": "three_sphere_buoys",
            "display_name": "AVBD/IPBF Three Sphere Buoys",
            "method": f"AVBD/IPBF {self.config.coupling_mode} fluid-solid coupling",
            "coordinate_system": {
                "source": "Newton Y-up right-handed",
                "blender_recommended_axis_conversion": "newton-y-up-to-blender-z-up",
                "blender_axis_mapping": "(x, y, z)_Newton -> (x, -z, y)_Blender",
            },
            "viewer_used": False,
            "offline_renderer": "Blender cache consumer",
            "fluid_particle_count": int(self.model.particle_count),
            "rigid_body_count": int(self.model.body_count),
            "dynamic_rigid_body_count": len(buoy_body_indices),
            "shape_count": int(self.model.shape_count),
            "boundary_sample_count": int(self.boundary_model.sample_count),
            "buoy_body_indices": buoy_body_indices,
            "body_labels": [f"fsi_sphere_buoy_{index}" for index in range(len(buoy_body_indices))],
            "body_masses": body_masses,
            "sphere_radii": [self.config.sphere_radius for _ in buoy_body_indices],
            "sphere_densities": list(self.config.sphere_densities),
            "tank": {
                "half_width": self.config.container_half_width,
                "half_depth": self.config.container_half_depth,
                "half_height": self.config.wall_half_height,
                "wall_thickness": self.config.wall_thickness,
                "floor_y": self.config.floor_y,
                "top_y": self.top_y,
            },
            "fluid": {
                "rest_density": self.config.rest_density,
                "particle_mass": self.config.particle_mass,
                "particle_radius": self.config.particle_radius,
                "smoothing_radius": self.config.smoothing_radius,
                "pool_dims": [self.config.pool_dim_x, self.config.pool_dim_y, self.config.pool_dim_z],
                "pool_bottom_clearance": self.config.pool_bottom_clearance,
                "initial_surface_y": self._compute_pool_surface_y(),
            },
            "solver": {
                "frame_dt": self.config.frame_dt,
                "fps": self.config.fps,
                "sim_dt": self.config.sim_dt,
                "sim_substeps": self.config.sim_substeps,
                "ipbf_iterations": self.config.ipbf_iterations,
                "rigid_iterations": self.config.rigid_iterations,
                "coupling_iterations": self.config.coupling_iterations,
                "coupling_mode": self.config.coupling_mode,
                "hydrostatic_volume_mode": self.config.hydrostatic_volume_mode,
                "include_static_boundary_samples": self.config.include_static_boundary_samples,
                "velocity_damping": self.config.velocity_damping,
                "fsi_projection_reaction_relaxation": self.config.fsi_projection_reaction_relaxation,
                "fsi_velocity_projection_reaction_relaxation": self.config.fsi_velocity_projection_reaction_relaxation,
                "fsi_pressure_reaction_relaxation": self.config.fsi_pressure_reaction_relaxation,
                "fsi_static_boundary_weight": self.config.fsi_static_boundary_weight,
            },
        }

    def _compute_pool_surface_y(self) -> float:
        _, pool_half_span_y, _ = _particle_grid_half_span(
            self.config.pool_dim_x,
            self.config.pool_dim_y,
            self.config.pool_dim_z,
            self.config.cell,
            self.config.cell,
            self.config.cell,
        )
        return self.floor_y + self.config.pool_bottom_clearance + 2.0 * pool_half_span_y + self.config.particle_radius

    def _add_container(self, builder: newton.ModelBuilder) -> None:
        wall_t = self.config.wall_thickness
        hx = self.config.container_half_width
        hz = self.config.container_half_depth
        hy = self.config.wall_half_height
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

    def _add_sphere_buoys(self, builder: newton.ModelBuilder) -> list[int]:
        water_surface_y = self._compute_pool_surface_y()
        sphere_center_y = water_surface_y + self.config.sphere_bottom_gap + self.config.sphere_radius
        x_positions = (-0.30, 0.0, 0.30)
        bodies: list[int] = []
        for index, (x_position, density) in enumerate(zip(x_positions, self.config.sphere_densities, strict=True)):
            body = builder.add_body(
                xform=wp.transform((x_position, sphere_center_y, 0.0), wp.quat_identity()),
                label=f"fsi_sphere_buoy_{index}",
            )
            builder.add_shape_sphere(
                body=body,
                radius=self.config.sphere_radius,
                cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.0),
            )
            bodies.append(body)
        return bodies

    def _add_pool(self, builder: newton.ModelBuilder) -> None:
        _, pool_half_span_y, _ = _particle_grid_half_span(
            self.config.pool_dim_x,
            self.config.pool_dim_y,
            self.config.pool_dim_z,
            self.config.cell,
            self.config.cell,
            self.config.cell,
        )
        pool_center = (
            0.0,
            self.floor_y + self.config.pool_bottom_clearance + pool_half_span_y,
            0.0,
        )
        builder.add_particle_grid(
            pos=_particle_grid_origin_from_center(
                pool_center,
                self.config.pool_dim_x,
                self.config.pool_dim_y,
                self.config.pool_dim_z,
                self.config.cell,
                self.config.cell,
                self.config.cell,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.config.pool_dim_x,
            dim_y=self.config.pool_dim_y,
            dim_z=self.config.pool_dim_z,
            cell_x=self.config.cell,
            cell_y=self.config.cell,
            cell_z=self.config.cell,
            mass=self.config.particle_mass,
            jitter=0.0,
            radius_mean=self.config.particle_radius,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["three_sphere_buoys"], default="three_sphere_buoys")
    parser.add_argument("--device", default="auto", help="Warp device, e.g. cuda:0, cpu, or auto.")
    parser.add_argument("--output-dir", type=Path, default=Path(".blender/cache/fsi_three_sphere_buoys"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing frame_*.npz files in output-dir.")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--record-frames", type=int, default=30)
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Simulation time [s] at which recording starts. Overrides --warmup-frames when provided.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Recorded video duration [s]. Overrides --record-frames when provided.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output cache/video frame rate [Hz]. Sets --record-stride from the engine FPS when provided.",
    )
    parser.add_argument(
        "--record-initial-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record the current state at the start time before advancing the first recorded frame.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--sim-substeps", type=int, default=6)
    parser.add_argument("--ipbf-iterations", type=int, default=2)
    parser.add_argument("--rigid-iterations", type=int, default=2)
    parser.add_argument("--coupling-iterations", type=int, default=3)
    parser.add_argument("--coupling-mode", choices=["interlinked", "loose"], default="interlinked")
    parser.add_argument("--pool-dims", type=int, nargs=3, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--cell", type=float, default=None)
    parser.add_argument("--sphere-radius", type=float, default=None)
    parser.add_argument("--sphere-bottom-gap", type=float, default=None)
    parser.add_argument("--sphere-densities", type=float, nargs=3, metavar=("LIGHT", "MID", "HEAVY"), default=None)
    parser.add_argument("--pressure-reaction-relaxation", type=float, default=2.0)
    parser.add_argument("--projection-reaction-relaxation", type=float, default=0.0)
    parser.add_argument("--velocity-reaction-relaxation", type=float, default=0.0)
    parser.add_argument(
        "--hydrostatic-volume-mode",
        choices=[
            "raw",
            "dynamic-shape-volume",
            "dynamic-shape-surface-quadrature",
            "dynamic-shape-surface-thickness",
        ],
        default="dynamic-shape-surface-quadrature",
    )
    parser.add_argument(
        "--include-static-boundary-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-velocities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-density", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-boundary-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compress-cache", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quiet:
        wp.config.quiet = True

    config = _config_from_args(args)
    schedule = _schedule_from_args(args, config)
    device = _resolve_device(args.device)
    scene = ThreeSphereBuoyScene(config, device)
    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    _prepare_output_dir(output_dir, frames_dir, overwrite=args.overwrite)

    if not args.quiet:
        print(f"Exporting {args.scene} on {scene.model.device}")
        print(f"Output: {output_dir}")
        print(f"Particles: {scene.model.particle_count}, bodies: {scene.model.body_count}")

    warmup_ms: list[float] = []
    for _ in range(schedule["warmup_frames"]):
        _synchronize(scene.model.device)
        warmup_start = time.perf_counter()
        scene.simulate_frame()
        _synchronize(scene.model.device)
        warmup_ms.append((time.perf_counter() - warmup_start) * 1000.0)

    stats = ExportStats(sim_ms=[], export_ms=[], warmup_ms=warmup_ms)
    for record_index in range(schedule["record_frames"]):
        if record_index > 0 or not schedule["record_initial_frame"]:
            for _ in range(schedule["record_stride"]):
                _synchronize(scene.model.device)
                sim_start = time.perf_counter()
                scene.simulate_frame()
                _synchronize(scene.model.device)
                stats.sim_ms.append((time.perf_counter() - sim_start) * 1000.0)
        elif stats.warmup_ms:
            _synchronize(scene.model.device)

        export_start = time.perf_counter()
        payload = scene.make_frame_payload(
            record_index=record_index,
            save_velocities=args.save_velocities,
            save_density=args.save_density,
            save_boundary_samples=args.save_boundary_samples,
        )
        _save_npz(frames_dir / f"frame_{record_index:04d}.npz", compressed=args.compress_cache, payload=payload)
        stats.export_ms.append((time.perf_counter() - export_start) * 1000.0)

        if not args.quiet and (record_index + 1) % max(1, args.log_interval) == 0:
            avg_ms = float(np.mean(np.asarray(stats.sim_ms, dtype=np.float64))) if stats.sim_ms else 0.0
            print(f"Recorded {record_index + 1}/{schedule['record_frames']} frames, avg sim {avg_ms:.2f} ms/frame")

    metadata = _build_metadata(
        scene,
        config=config,
        args=args,
        schedule=schedule,
        stats=stats,
    )
    _write_json(output_dir / "metadata.json", metadata)
    if not args.quiet:
        timing = metadata["timing"]
        print(f"Done. Newton simulation: {timing['avg_sim_ms_per_frame']:.2f} ms/frame")
        print(f"Cache export: {timing['avg_export_ms_per_recorded_frame']:.2f} ms/recorded frame")


def _config_from_args(args: argparse.Namespace) -> SceneConfig:
    config = SceneConfig(
        fps=args.fps,
        sim_substeps=args.sim_substeps,
        ipbf_iterations=args.ipbf_iterations,
        rigid_iterations=args.rigid_iterations,
        coupling_iterations=args.coupling_iterations,
        coupling_mode=args.coupling_mode,
        hydrostatic_volume_mode=args.hydrostatic_volume_mode,
        include_static_boundary_samples=args.include_static_boundary_samples,
        fsi_projection_reaction_relaxation=args.projection_reaction_relaxation,
        fsi_velocity_projection_reaction_relaxation=args.velocity_reaction_relaxation,
        fsi_pressure_reaction_relaxation=args.pressure_reaction_relaxation,
    )
    if args.pool_dims is not None:
        config = replace(
            config,
            pool_dim_x=int(args.pool_dims[0]),
            pool_dim_y=int(args.pool_dims[1]),
            pool_dim_z=int(args.pool_dims[2]),
        )
    if args.cell is not None:
        config = replace(config, cell=float(args.cell), boundary_spacing=float(args.cell))
    if args.sphere_radius is not None:
        config = replace(config, sphere_radius=float(args.sphere_radius))
    if args.sphere_bottom_gap is not None:
        config = replace(config, sphere_bottom_gap=float(args.sphere_bottom_gap))
    if args.sphere_densities is not None:
        config = replace(config, sphere_densities=tuple(float(value) for value in args.sphere_densities))
    return config


def _schedule_from_args(args: argparse.Namespace, config: SceneConfig) -> dict[str, Any]:
    warmup_frames = max(0, int(args.warmup_frames))
    record_frames = max(0, int(args.record_frames))
    record_stride = max(1, int(args.record_stride))
    requested_video_fps = float(args.video_fps) if args.video_fps is not None else None

    if args.start_time is not None:
        warmup_frames = max(0, int(round(float(args.start_time) * float(config.fps))))
    if requested_video_fps is not None:
        record_stride = max(1, int(round(float(config.fps) / max(requested_video_fps, 1.0e-12))))
    actual_video_fps = float(config.fps) / float(record_stride)
    if args.duration is not None:
        video_fps = requested_video_fps if requested_video_fps is not None else actual_video_fps
        record_frames = max(0, int(round(float(args.duration) * max(video_fps, 1.0e-12))))

    return {
        "warmup_frames": warmup_frames,
        "start_time": warmup_frames * config.frame_dt,
        "record_frames": record_frames,
        "record_stride": record_stride,
        "record_initial_frame": bool(args.record_initial_frame),
        "requested_duration": None if args.duration is None else float(args.duration),
        "requested_video_fps": requested_video_fps,
        "actual_video_fps": actual_video_fps,
        "recorded_video_duration": record_frames / actual_video_fps if actual_video_fps > 0.0 else 0.0,
    }


def _resolve_device(device_name: str) -> wp.context.Device:
    if device_name != "auto":
        return wp.get_device(device_name)
    try:
        return wp.get_device("cuda:0")
    except Exception:
        return wp.get_device("cpu")


def _parse_hydrostatic_volume_mode(mode: str) -> FSIBoundaryModel.HydrostaticVolumeMode:
    mode_map = {
        "raw": FSIBoundaryModel.HydrostaticVolumeMode.NONE,
        "dynamic-shape-volume": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_VOLUME,
        "dynamic-shape-surface-quadrature": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_QUADRATURE,
        "dynamic-shape-surface-thickness": FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_SHAPE_SURFACE_THICKNESS,
    }
    return mode_map[mode]


def _parse_coupling_mode(mode: str) -> SolverFSI.Config.CouplingMode:
    if mode == "interlinked":
        return SolverFSI.Config.CouplingMode.INTERLINKED
    if mode == "loose":
        return SolverFSI.Config.CouplingMode.LOOSE
    raise ValueError(f"Unsupported coupling mode: {mode}")


def _synchronize(device: wp.context.Device) -> None:
    wp.synchronize_device(device)


def _prepare_output_dir(output_dir: Path, frames_dir: Path, *, overwrite: bool) -> None:
    if frames_dir.exists() and any(frames_dir.glob("frame_*.npz")):
        if not overwrite:
            raise FileExistsError(f"{frames_dir} already contains frame caches. Pass --overwrite to replace them.")
        for path in frames_dir.glob("frame_*.npz"):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)


def _save_npz(path: Path, *, compressed: bool, payload: dict[str, np.ndarray]) -> None:
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _build_metadata(
    scene: ThreeSphereBuoyScene,
    *,
    config: SceneConfig,
    args: argparse.Namespace,
    schedule: dict[str, Any],
    stats: ExportStats,
) -> dict[str, Any]:
    metadata = scene.metadata_base()
    timing = stats.timing_summary(config.sim_substeps, schedule["warmup_frames"])
    metadata.update(
        {
            "cache_schema_version": 1,
            "newton_version": getattr(newton, "__version__", "unknown"),
            "warp_version": getattr(wp, "__version__", "unknown"),
            "device": str(scene.model.device),
            "scene_config": _jsonify(asdict(config)),
            "export": {
                "output_dir": str(args.output_dir),
                "warmup_frames": schedule["warmup_frames"],
                "start_time": schedule["start_time"],
                "record_frames": schedule["record_frames"],
                "record_stride": schedule["record_stride"],
                "record_initial_frame": schedule["record_initial_frame"],
                "requested_duration": schedule["requested_duration"],
                "requested_video_fps": schedule["requested_video_fps"],
                "actual_video_fps": schedule["actual_video_fps"],
                "recorded_video_duration": schedule["recorded_video_duration"],
                "simulated_record_frames": max(
                    0,
                    schedule["record_frames"] - (1 if schedule["record_initial_frame"] else 0),
                )
                * schedule["record_stride"],
                "save_velocities": bool(args.save_velocities),
                "save_density": bool(args.save_density),
                "save_boundary_samples": bool(args.save_boundary_samples),
                "compress_cache": bool(args.compress_cache),
            },
            "timing": timing,
        }
    )
    metadata["title_card_lines"] = _title_card_lines(scene, config, timing, schedule)
    return metadata


def _title_card_lines(
    scene: ThreeSphereBuoyScene,
    config: SceneConfig,
    timing: dict[str, Any],
    schedule: dict[str, Any],
) -> list[str]:
    sim_hz = 1.0 / max(config.sim_dt, 1.0e-12)
    return [
        "AVBD/IPBF Fluid-Solid Coupling",
        f"{_format_count(scene.model.particle_count)} Particles | {scene.model.body_count} Rigid Bodies",
        (
            f"h = 1/{int(round(sim_hz))} s | frame dt = 1/{config.fps} s | "
            f"video = {schedule['actual_video_fps']:.1f} fps"
        ),
        (
            f"IPBF {config.ipbf_iterations} iters | AVBD {config.rigid_iterations} iters | "
            f"FSI {config.coupling_iterations} coupling iters"
        ),
        f"{timing['avg_sim_ms_per_frame']:.1f} ms/frame in Newton, no viewer/render",
    ]


def _format_count(count: int) -> str:
    if count >= 1_000_000:
        value = count / 1_000_000.0
        return f"{value:.1f}M" if value < 10.0 else f"{value:.0f}M"
    if count >= 1_000:
        value = count / 1_000.0
        return f"{value:.1f}K" if value < 10.0 else f"{value:.0f}K"
    return str(count)


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _particle_grid_half_span(
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
) -> tuple[float, float, float]:
    return (
        0.5 * float(dim_x - 1) * cell_x,
        0.5 * float(dim_y - 1) * cell_y,
        0.5 * float(dim_z - 1) * cell_z,
    )


def _particle_grid_origin_from_center(
    center: tuple[float, float, float],
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
) -> wp.vec3:
    hx, hy, hz = _particle_grid_half_span(dim_x, dim_y, dim_z, cell_x, cell_y, cell_z)
    cx, cy, cz = center
    return wp.vec3(cx - hx, cy - hy, cz - hz)


if __name__ == "__main__":
    main()
