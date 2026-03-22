# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IPBF Box Container Boundary Particles
#
# Experimental open-top box example that enables the IPBF boundary-particle
# mode. Static box walls are sampled into boundary particles, while the
# existing shape-contact projection can optionally remain active as a fallback
# anti-tunneling mechanism.
#
# Command: python -m newton.examples ipbf_box_container_boundary_particles
#
###########################################################################

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverIPBF


@wp.kernel
def scale_velocities(
    particle_qd: wp.array(dtype=wp.vec3),
    scale: float,
):
    """Apply a uniform multiplicative velocity damping."""
    tid = wp.tid()
    particle_qd[tid] = scale * particle_qd[tid]


def build_box_wireframe(
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    device,
) -> tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.vec3)]:
    """Build a box wireframe for viewer line rendering."""
    corners = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float32,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    starts = wp.array(corners[[i for i, _ in edges]], dtype=wp.vec3, device=device)
    ends = wp.array(corners[[j for _, j in edges]], dtype=wp.vec3, device=device)
    return starts, ends


class Example:
    """Experimental closed-box IPBF container using boundary particles."""

    class DenseLevel:
        """Named particle-density presets for the box container example."""

        NORMAL = "normal"
        DENSE = "dense"
        X_DENSE = "x-dense"
        LARGE = "large"

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args
        self._reset_key_prev = False
        self._capture_dirty = False

        self.container_half_width = 0.55
        self.container_half_depth = 0.55
        self.wall_thickness = 0.05
        self.wall_half_height = 0.45
        self.floor_y = 0.0
        self.top_y = 2.0 * self.wall_half_height
        self.boundary_spacing = 0.05
        # A small global damping still helps the current prototype settle,
        # even when boundary particles provide near-wall density support.
        self.velocity_damping = 0.99
        self.use_shape_contacts = bool(getattr(args, "use_shape_contacts", True))
        default_dense_level = self.DenseLevel.NORMAL if bool(getattr(args, "test", False)) else self.DenseLevel.LARGE
        self.dense_level = str(getattr(args, "dense_level", default_dense_level))
        self.initial_viscosity_coefficient = float(getattr(args, "viscosity_coefficient", 0.005))
        self.initial_xsph_coefficient = float(getattr(args, "xsph_coefficient", 0.02))
        particle_block = self._get_particle_block_config(self.dense_level)
        self.initial_particle_velocity = particle_block["velocity"]
        self.boundary_spacing = particle_block["boundary_spacing"]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.default_shape_cfg.mu = 0.0
        self._add_container(builder)

        builder.add_particle_grid(
            pos=particle_block["pos"],
            rot=wp.quat_identity(),
            vel=self.initial_particle_velocity,
            dim_x=particle_block["dim_x"],
            dim_y=particle_block["dim_y"],
            dim_z=particle_block["dim_z"],
            cell_x=particle_block["cell_x"],
            cell_y=particle_block["cell_y"],
            cell_z=particle_block["cell_z"],
            mass=particle_block["mass"],
            jitter=0.0,
            radius_mean=particle_block["radius_mean"],
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.81, 0.0))

        self.solver = SolverIPBF(
            self.model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=particle_block["smoothing_radius"],
                boundary_mode=SolverIPBF.Config.BoundaryMode.BOUNDARY_PARTICLES,
                iterations=particle_block["iterations"],
                relaxation=0.5,
                viscosity_coefficient=self.initial_viscosity_coefficient,
                xsph_coefficient=self.initial_xsph_coefficient,
            ),
        )
        self.solver.setup_boundary_particles_box(
            half_width=self.container_half_width,
            half_depth=self.container_half_depth,
            wall_half_height=self.wall_half_height,
            spacing=self.boundary_spacing,
            include_top=True,
        )

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=0.06,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.15, 0.55, 1.0),
            dtype=wp.vec3,
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
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(1.85, 1.55, 1.85),
            pitch=-32.0,
            yaw=-135.0,
        )

        self.reset()
        self.capture()

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
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 2.0 * hy + wall_t, 0.0), wp.quat_identity()),
            hx=hx + wall_t,
            hy=wall_t,
            hz=hz + wall_t,
        )

    def _get_particle_block_config(self, dense_level: str) -> dict[str, object]:
        """Return the particle-block preset for the requested density level."""
        presets = {
            self.DenseLevel.NORMAL: {
                "pos": wp.vec3(-0.2, 0.12, -0.2),
                "dim_x": 6,
                "dim_y": 7,
                "dim_z": 6,
                "cell_x": 0.075,
                "cell_y": 0.075,
                "cell_z": 0.075,
                "mass": 0.5,
                "radius_mean": 0.03,
                "smoothing_radius": 0.12,
                "iterations": 5,
                "velocity": wp.vec3(0.3, 0.0, 0.1125),
                "boundary_spacing": 0.05,
            },
            self.DenseLevel.DENSE: {
                "pos": wp.vec3(-0.225, 0.12, -0.225),
                "dim_x": 7,
                "dim_y": 8,
                "dim_z": 7,
                "cell_x": 0.075,
                "cell_y": 0.075,
                "cell_z": 0.075,
                "mass": 0.5,
                "radius_mean": 0.03,
                "smoothing_radius": 0.12,
                "iterations": 5,
                "velocity": wp.vec3(0.3, 0.0, 0.1125),
                "boundary_spacing": 0.05,
            },
            self.DenseLevel.X_DENSE: {
                "pos": wp.vec3(-0.2625, 0.12, -0.2625),
                "dim_x": 8,
                "dim_y": 9,
                "dim_z": 8,
                "cell_x": 0.075,
                "cell_y": 0.075,
                "cell_z": 0.075,
                "mass": 0.5,
                "radius_mean": 0.03,
                "smoothing_radius": 0.12,
                "iterations": 5,
                "velocity": wp.vec3(0.3, 0.0, 0.1125),
                "boundary_spacing": 0.05,
            },
            self.DenseLevel.LARGE: {
                "pos": wp.vec3(-0.434, 0.035, -0.434),
                "dim_x": 32,
                "dim_y": 30,
                "dim_z": 32,
                "cell_x": 0.028,
                "cell_y": 0.028,
                "cell_z": 0.028,
                "mass": 0.022,
                "radius_mean": 0.012,
                "smoothing_radius": 0.048,
                "iterations": 6,
                "velocity": wp.vec3(0.15, 0.0, 0.06),
                "boundary_spacing": 0.03,
            },
        }
        if dense_level not in presets:
            raise ValueError(f"Unknown dense level: {dense_level}")
        return presets[dense_level]

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Dense Level: {self.dense_level}")
        changed, value = ui.checkbox("Use Shape Contacts", self.use_shape_contacts)
        if changed:
            self.set_use_shape_contacts(value)
        changed, value = ui.slider_float("Viscosity Coefficient", self.solver.viscosity_coefficient, 0.0, 0.1)
        if changed:
            self.set_viscosity_coefficient(value)
        changed, value = ui.slider_float("XSPH Coefficient", self.solver.xsph_coefficient, 0.0, 0.1)
        if changed:
            self.set_xsph_coefficient(value)

    def _mark_capture_dirty(self) -> None:
        self._capture_dirty = True
        self.graph = None

    def set_use_shape_contacts(self, enabled: bool) -> None:
        self.use_shape_contacts = bool(enabled)
        self._mark_capture_dirty()

    def set_viscosity_coefficient(self, coefficient: float) -> None:
        self.solver.viscosity_coefficient = float(coefficient)
        self._mark_capture_dirty()

    def set_xsph_coefficient(self, coefficient: float) -> None:
        self.solver.xsph_coefficient = float(coefficient)
        self._mark_capture_dirty()

    def reset(self):
        self.sim_time = 0.0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self.contacts.clear()
        self.viewer._paused = True

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
        self._capture_dirty = False

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
            contacts = self.contacts if self.use_shape_contacts else None
            self.solver.step(self.state_0, self.state_1, control=None, contacts=contacts, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            reset_down = bool(self.viewer.is_key_down("r"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        if self._capture_dirty:
            self.capture()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_radius = self.model.particle_radius.numpy()
        particle_speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)

        max_x = np.max(np.abs(particle_q[:, 0]) + particle_radius)
        max_z = np.max(np.abs(particle_q[:, 2]) + particle_radius)
        min_y = np.min(particle_q[:, 1] - particle_radius)
        max_y = np.max(particle_q[:, 1] + particle_radius)
        max_speed = np.max(particle_speed)

        assert max_x <= self.container_half_width + 0.02, f"particles escaped along x: {max_x:.3f}"
        assert max_z <= self.container_half_depth + 0.02, f"particles escaped along z: {max_z:.3f}"
        assert min_y >= -0.02, f"particles penetrated the floor: {min_y:.3f}"
        assert max_y <= self.top_y + 0.02, f"particles penetrated the ceiling: {max_y:.3f}"
        assert max_speed <= 2.0, f"particles retained excessive speed: {max_speed:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_lines(
            "/ipbf/container_wireframe",
            self.box_wire_starts,
            self.box_wire_ends,
            colors=(0.88, 0.9, 0.95),
            width=0.01,
        )
        self.viewer.log_points(
            "/ipbf/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--dense-level",
        type=str,
        choices=[Example.DenseLevel.NORMAL, Example.DenseLevel.DENSE, Example.DenseLevel.X_DENSE, Example.DenseLevel.LARGE],
        default=Example.DenseLevel.LARGE,
        help="Particle-density preset used to build the initial fluid block.",
    )
    parser.add_argument(
        "--use-shape-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable static shape contacts as a fallback anti-tunneling mechanism.",
    )
    parser.add_argument(
        "--viscosity-coefficient",
        type=float,
        default=0.005,
        help="SPH viscosity diffusion coefficient.",
    )
    parser.add_argument(
        "--xsph-coefficient",
        type=float,
        default=0.02,
        help="XSPH velocity smoothing coefficient.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
