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

import argparse
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipbf.ipbf_kernels import kernel_gradient, kernel_hessian, kernel_value
from newton.examples.ipbf.example_ipbf_box_container import Example as ExampleIPBFBoxContainer
from newton.solvers import FSIBoundaryModel, SolverIPBF
from newton.tests.unittest_utils import add_function_test, get_test_devices

KernelFamily = SolverIPBF.Config.KernelFamily


def kernel_density_contribution(
    mass: float,
    support_radius: float,
    distance: float,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> float:
    """Reference scalar kernel density contribution used by the IPBF kernels."""
    if support_radius <= 0.0 or distance >= support_radius:
        return 0.0

    if kernel_family == KernelFamily.POLY6:
        support_radius2 = support_radius * support_radius
        x = support_radius2 - distance * distance
        value = 315.0 / (64.0 * np.pi * support_radius**9) * x**3
    else:
        q = 2.0 * distance / support_radius
        normalization = 8.0 / (np.pi * support_radius**3)
        if q < 1.0:
            value = normalization * (1.0 - 1.5 * q**2 + 0.75 * q**3)
        else:
            value = normalization * 0.25 * (2.0 - q) ** 3

    return float(mass * value)


def kernel_gradient_contribution(
    mass: float,
    support_radius: float,
    displacement: np.ndarray,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> np.ndarray:
    """Reference spatial kernel gradient contribution used by the IPBF kernels."""
    distance2 = float(np.dot(displacement, displacement))
    if support_radius <= 0.0 or distance2 >= support_radius * support_radius or distance2 == 0.0:
        return np.zeros(3, dtype=np.float32)

    if kernel_family == KernelFamily.POLY6:
        x = support_radius * support_radius - distance2
        scale = -945.0 / (32.0 * np.pi * support_radius**9) * x * x
        return (mass * scale * displacement).astype(np.float32)

    distance = float(np.sqrt(distance2))
    q = 2.0 * distance / support_radius
    normalization = 8.0 / (np.pi * support_radius**3)
    q_scale = 2.0 / support_radius

    if q < 1.0:
        dW_dr = normalization * q_scale * (-3.0 * q + 2.25 * q**2)
    else:
        dW_dr = normalization * q_scale * (-0.75 * (2.0 - q) ** 2)

    return (mass * (dW_dr / distance) * displacement).astype(np.float32)


def kernel_hessian_reference(
    support_radius: float,
    displacement: np.ndarray,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> np.ndarray:
    """Reference spatial kernel Hessian used by the IPBF kernels."""
    distance2 = float(np.dot(displacement, displacement))
    if support_radius <= 0.0 or distance2 >= support_radius * support_radius:
        return np.zeros((3, 3), dtype=np.float32)

    if kernel_family == KernelFamily.POLY6:
        x = support_radius * support_radius - distance2
        identity = np.eye(3, dtype=np.float32)
        outer = np.outer(displacement, displacement)
        return (
            -945.0 / (32.0 * np.pi * support_radius**9) * x * x * identity
            + 945.0 / (8.0 * np.pi * support_radius**9) * x * outer
        ).astype(np.float32)

    normalization = 8.0 / (np.pi * support_radius**3)
    q_scale = 2.0 / support_radius
    q_scale2 = q_scale * q_scale
    identity = np.eye(3, dtype=np.float32)

    if distance2 == 0.0:
        return (normalization * q_scale2 * (-3.0) * identity).astype(np.float32)

    distance = float(np.sqrt(distance2))
    q = 2.0 * distance / support_radius

    if q < 1.0:
        dW_dr = normalization * q_scale * (-3.0 * q + 2.25 * q**2)
        d2W_dr2 = normalization * q_scale2 * (-3.0 + 4.5 * q)
    else:
        dW_dr = normalization * q_scale * (-0.75 * (2.0 - q) ** 2)
        d2W_dr2 = normalization * q_scale2 * (1.5 * (2.0 - q))

    inv_r = 1.0 / distance
    inv_r2 = inv_r * inv_r
    outer = np.outer(displacement, displacement)
    return (dW_dr * inv_r * identity + (d2W_dr2 - dW_dr * inv_r) * inv_r2 * outer).astype(np.float32)


def expected_constraint_hessian(
    masses: list[float],
    support_radius: float,
    displacements: list[np.ndarray],
    rest_density: float,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> np.ndarray:
    """Reference constraint Hessian built from kernel Hessian contributions."""
    if rest_density <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)

    hessian = np.zeros((3, 3), dtype=np.float32)
    for mass, displacement in zip(masses, displacements, strict=True):
        hessian += mass * kernel_hessian_reference(support_radius, displacement, kernel_family)

    return (hessian / rest_density).astype(np.float32)


def expected_neighbor_constraint_hessian(
    mass_self: float,
    support_radius: float,
    displacement_neighbor_minus_self: np.ndarray,
    rest_density: float,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> np.ndarray:
    """Reference neighbor constraint Hessian with respect to the current particle."""
    if rest_density <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)

    return (
        mass_self
        * kernel_hessian_reference(support_radius, displacement_neighbor_minus_self, kernel_family)
        / rest_density
    ).astype(np.float32)


def expected_ipbf_hessian(
    gradient: np.ndarray,
    mass: float,
    compliance: float,
    dt: float,
    regularization: float,
    constraint: float = 0.0,
    constraint_hessian: np.ndarray | None = None,
    neighbor_gradients: list[np.ndarray] | None = None,
    neighbor_constraints: list[float] | None = None,
    neighbor_constraint_hessians: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Reference IPBF Hessian used by the current skeleton."""
    inertia_scale = 0.0 if dt <= 0.0 else compliance * mass / (dt * dt)
    hessian = ((inertia_scale + regularization) * np.eye(3, dtype=np.float32) + np.outer(gradient, gradient)).astype(
        np.float32
    )

    if neighbor_gradients is not None:
        for neighbor_gradient in neighbor_gradients:
            hessian += np.outer(neighbor_gradient, neighbor_gradient).astype(np.float32)

    if constraint_hessian is not None and constraint != 0.0:
        hessian += abs(constraint) * np.diag(np.linalg.norm(constraint_hessian, axis=0).astype(np.float32))

    if neighbor_constraints is not None and neighbor_constraint_hessians is not None:
        for neighbor_constraint, neighbor_constraint_hessian in zip(
            neighbor_constraints, neighbor_constraint_hessians, strict=True
        ):
            if neighbor_constraint != 0.0:
                hessian += abs(neighbor_constraint) * np.diag(
                    np.linalg.norm(neighbor_constraint_hessian, axis=0).astype(np.float32)
                )

    return hessian.astype(np.float32)


def expected_ipbf_delta(force: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Reference local IPBF position update used by the current skeleton."""
    if abs(float(np.linalg.det(hessian))) <= 1.0e-8:
        return np.zeros(3, dtype=np.float32)

    return np.linalg.solve(hessian, force).astype(np.float32)


def expected_two_particle_force(
    constraint_self: float,
    gradient_self: np.ndarray,
    constraint_neighbor: float,
    support_radius: float,
    displacement_neighbor_minus_self: np.ndarray,
    rest_density: float,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
) -> np.ndarray:
    """Reference two-particle IPBF force with neighbor constraint contributions."""
    if rest_density <= 0.0:
        return np.zeros(3, dtype=np.float32)

    neighbor_term = (
        constraint_neighbor
        * kernel_gradient_contribution(1.0, support_radius, displacement_neighbor_minus_self, kernel_family)
        / rest_density
    )
    return (-constraint_self * gradient_self + neighbor_term).astype(np.float32)


@wp.kernel
def evaluate_kernel_interface(
    displacement: wp.array(dtype=wp.vec3),
    support_radius: float,
    kernel_family: int,
    value: wp.array(dtype=float),
    gradient: wp.array(dtype=wp.vec3),
    hessian: wp.array(dtype=wp.mat33),
):
    """Evaluate the current kernel interface for a set of query displacements."""
    tid = wp.tid()
    disp = displacement[tid]
    value[tid] = kernel_value(wp.dot(disp, disp), support_radius, kernel_family)
    gradient[tid] = kernel_gradient(disp, support_radius, kernel_family)
    hessian[tid] = kernel_hessian(disp, support_radius, kernel_family)


def run_two_particle_ipbf_step(
    device,
    *,
    iterations: int,
    relaxation: float,
    rest_density: float = 1.0,
    support_radius: float = 0.3,
    regularization: float = 1.0e-6,
    kernel_family: int | KernelFamily = KernelFamily.CUBIC_SPLINE,
    damping_compliance: float = 1.0 / 1000.0,
    damping_beta: float = 0.0,
    initial_velocities: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    xsph_coefficient: float = 0.0,
    viscosity_coefficient: float = 0.0,
):
    """Run one zero-gravity IPBF step for a symmetric two-particle setup."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(*initial_velocities[0]),
            wp.vec3(*initial_velocities[1]),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity(gravity)

    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            kernel_family=kernel_family,
            hessian_regularization=regularization,
            iterations=iterations,
            relaxation=relaxation,
            use_constraint_clamp=False,
            damping_compliance=damping_compliance,
            damping_beta=damping_beta,
            xsph_coefficient=xsph_coefficient,
            viscosity_coefficient=viscosity_coefficient,
        ),
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)
    return state_1


def run_single_particle_ground_step(device, *, use_contacts: bool):
    """Run one gravity step for a particle above a ground plane."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 0.06, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=1.0,
        radius=0.05,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.2,
            iterations=0,
        ),
    )

    contacts = None
    if use_contacts:
        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
        contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=contacts, dt=0.1)
    return state_1, contacts


def run_boundary_density_step(
    device,
    *,
    use_boundary_model: bool,
    boundary_spacing: float = 0.25,
    boundary_body: str = "static",
    static_boundary_weight: float = 1.0,
    hydrostatic_volume_mode: FSIBoundaryModel.HydrostaticVolumeMode = FSIBoundaryModel.HydrostaticVolumeMode.NONE,
):
    """Run a zero-iteration IPBF step near a sampled box boundary."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 0.5, 0.0),
            wp.vec3(0.05, 0.5, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )
    box_body = -1
    if boundary_body == "dynamic":
        box_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
    elif boundary_body != "static":
        raise ValueError(f"Unsupported boundary_body: {boundary_body}")

    builder.add_shape_box(
        body=box_body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.25,
        hy=0.25,
        hz=0.25,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = None
    if use_boundary_model:
        boundary_model = FSIBoundaryModel(
            model,
            spacing=boundary_spacing,
            support_radius=0.6,
            hydrostatic_volume_mode=hydrostatic_volume_mode,
            include_static=boundary_body == "static",
            include_dynamic=boundary_body == "dynamic",
            device=device,
        )

    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.6,
            hessian_regularization=1.0e-6,
            iterations=0,
            use_constraint_clamp=False,
            fsi_static_boundary_weight=static_boundary_weight,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)

    return state_1, solver


def run_static_boundary_viscosity_step(
    device,
    *,
    viscosity_boundary_coefficient: float,
    static_boundary_weight: float = 1.0,
):
    """Run one step with particles moving tangentially near a sampled static boundary."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 0.5, 0.0),
            wp.vec3(0.05, 0.5, 0.0),
        ],
        vel=[
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.25,
        hy=0.25,
        hz=0.25,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=0.25,
        support_radius=0.6,
        include_static=True,
        include_dynamic=False,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.6,
            hessian_regularization=1.0e-6,
            iterations=0,
            use_constraint_clamp=False,
            viscosity_coefficient=0.0,
            viscosity_boundary_coefficient=viscosity_boundary_coefficient,
            fsi_static_boundary_weight=static_boundary_weight,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)

    return state_1


def run_dynamic_box_pressure_reaction_step(device, *, reaction_relaxation: float):
    """Run one IPBF pressure iteration near a sampled dynamic box boundary."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 0.12, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=1.0,
        radius=0.02,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(
        body=body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.2,
        hy=0.1,
        hz=0.2,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(model, spacing=0.2, support_radius=0.4, device=device)
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.4,
            iterations=1,
            relaxation=1.0,
            fsi_projection_reaction_relaxation=0.0,
            fsi_velocity_projection_reaction_relaxation=0.0,
            fsi_pressure_reaction_relaxation=reaction_relaxation,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.1)

    return state_0, state_1, solver, boundary_model, body


def run_dynamic_box_projection_reaction_step(
    device,
    *,
    projection_reaction_relaxation: float | None = 1.0,
    legacy_reaction_relaxation: float | None = None,
):
    """Run a penetrating particle against a sampled dynamic box boundary."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    particle_mass = 2.0
    builder.add_particle(
        pos=wp.vec3(0.0, 0.12, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=particle_mass,
        radius=0.05,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(
        body=body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.25,
        hy=0.1,
        hz=0.25,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(model, spacing=0.25, support_radius=0.3, device=device)
    config_kwargs: dict[str, float | int | bool | SolverIPBF.Config.KernelFamily | None] = {
        "smoothing_radius": 0.3,
        "iterations": 0,
    }
    if legacy_reaction_relaxation is not None:
        config_kwargs["fsi_reaction_relaxation"] = legacy_reaction_relaxation
    else:
        config_kwargs["fsi_projection_reaction_relaxation"] = projection_reaction_relaxation

    solver = SolverIPBF(model, SolverIPBF.Config(**config_kwargs), boundary_model=boundary_model)

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.1
    solver.step(state_0, state_1, control=None, contacts=contacts, dt=dt)

    return state_0, state_1, solver, boundary_model, contacts, body, particle_mass, dt


def run_dynamic_box_velocity_reaction_step(
    device,
    *,
    velocity_reaction_relaxation: float | None = 1.0,
    projection_reaction_relaxation: float | None = 0.0,
    legacy_reaction_relaxation: float | None = None,
):
    """Run an approaching particle that is stopped by boundary velocity projection."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    particle_mass = 1.0
    builder.add_particle(
        pos=wp.vec3(-0.13, 0.0, 0.0),
        vel=wp.vec3(1.0, 0.0, 0.0),
        mass=particle_mass,
        radius=0.02,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(
        body=body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.1,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(model, spacing=0.1, support_radius=0.2, device=device)
    config_kwargs: dict[str, float | int | bool | SolverIPBF.Config.KernelFamily | None] = {
        "smoothing_radius": 0.2,
        "iterations": 0,
        "boundary_velocity_damping": 1.0,
    }
    if legacy_reaction_relaxation is not None:
        config_kwargs["fsi_reaction_relaxation"] = legacy_reaction_relaxation
    else:
        config_kwargs["fsi_projection_reaction_relaxation"] = projection_reaction_relaxation
        config_kwargs["fsi_velocity_projection_reaction_relaxation"] = velocity_reaction_relaxation

    solver = SolverIPBF(model, SolverIPBF.Config(**config_kwargs), boundary_model=boundary_model)

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    dt = 0.005
    solver.step(state_0, state_1, control=None, contacts=contacts, dt=dt)

    return state_1, boundary_model, contacts, body


def run_single_particle_ground_rollout(
    device,
    *,
    steps: int,
    boundary_velocity_damping: float = 1.0,
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    dt: float = 0.1,
):
    """Run several gravity steps for a particle above a ground plane."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 0.06, 0.0),
        vel=wp.vec3(*initial_velocity),
        mass=1.0,
        radius=0.05,
    )
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity(gravity)
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.2,
            iterations=0,
            boundary_velocity_damping=boundary_velocity_damping,
        ),
    )

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = model.contacts(collision_pipeline=collision_pipeline)
    state_0 = model.state()
    state_1 = model.state()

    positions = []
    velocities = []
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=contacts, dt=dt)
        state_0, state_1 = state_1, state_0
        positions.append(state_0.particle_q.numpy()[0].copy())
        velocities.append(state_0.particle_qd.numpy()[0].copy())

    return state_0, contacts, np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)


def run_ipbf_box_container_rollout(device, *, num_frames: int):
    """Run the public IPBF box-container example with a null viewer."""
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull()
        example = ExampleIPBFBoxContainer(viewer, args=argparse.Namespace(test=True))
        example.graph = None

        speed_history = []
        max_abs_x = 0.0
        max_abs_z = 0.0
        min_y = np.inf

        for _ in range(num_frames):
            example.step()
            particle_q = example.state_0.particle_q.numpy()
            particle_qd = example.state_0.particle_qd.numpy()
            particle_radius = example.model.particle_radius.numpy()

            max_abs_x = max(max_abs_x, float(np.max(np.abs(particle_q[:, 0]) + particle_radius)))
            max_abs_z = max(max_abs_z, float(np.max(np.abs(particle_q[:, 2]) + particle_radius)))
            min_y = min(min_y, float(np.min(particle_q[:, 1] - particle_radius)))
            speed_history.append(float(np.linalg.norm(particle_qd, axis=1).max()))

    return np.asarray(speed_history, dtype=np.float32), max_abs_x, max_abs_z, float(min_y)


def test_ipbf_registers_attributes_and_applies_inertial_prediction(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particle(
        pos=wp.vec3(0.0, 1.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=2.0,
        radius=0.1,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    config = SolverIPBF.Config(
        rest_density=1234.0,
        smoothing_radius=0.25,
        compliance=0.125,
        hessian_regularization=1.0e-6,
    )
    solver = SolverIPBF(model, config)

    state_0 = model.state()
    state_1 = model.state()

    dt = 0.1
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    test.assertTrue(hasattr(model, "ipbf"))
    test.assertTrue(hasattr(state_1, "ipbf"))

    test.assertAlmostEqual(float(model.ipbf.rest_density.numpy()[0]), config.rest_density, places=6)
    test.assertAlmostEqual(float(model.ipbf.smoothing_radius.numpy()[0]), config.smoothing_radius, places=6)
    test.assertAlmostEqual(float(model.ipbf.compliance.numpy()[0]), config.compliance, places=6)

    expected_y = np.array([0.0, 1.0 - 9.81 * dt * dt, 0.0], dtype=np.float32)
    expected_v = np.array([0.0, -9.81 * dt, 0.0], dtype=np.float32)
    expected_density = kernel_density_contribution(2.0, config.smoothing_radius, 0.0)
    expected_hessian = expected_ipbf_hessian(
        np.zeros(3, dtype=np.float32),
        mass=2.0,
        compliance=config.compliance,
        dt=dt,
        regularization=config.hessian_regularization,
    )

    np.testing.assert_allclose(state_1.ipbf.y.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        state_1.ipbf.density.numpy(), np.array([expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(state_1.ipbf.constraint.numpy(), np.zeros(1, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        state_1.ipbf.constraint_gradient.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(state_1.ipbf.force.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.hessian.numpy()[0], expected_hessian, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy(), np.zeros((1, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_ipbf_requires_registered_attributes(test, device):
    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        mass=1.0,
        radius=0.1,
    )
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "requires IPBF custom attributes"):
        SolverIPBF(model)


def test_ipbf_multi_particle_shell_step_builds_neighbor_search(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            smoothing_radius=0.2,
            iterations=0,
        ),
    )

    test.assertIsNotNone(model.particle_grid)

    state_0 = model.state()
    state_1 = model.state()

    dt = 0.05
    for _ in range(3):
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    q = state_0.particle_q.numpy()
    qd = state_0.particle_qd.numpy()
    density = state_0.ipbf.density.numpy()
    neighbor_count = state_0.ipbf.neighbor_count.numpy()

    test.assertEqual(q.shape, (2, 3))
    test.assertEqual(qd.shape, (2, 3))
    test.assertLess(q[0, 1], 1.0)
    test.assertLess(q[1, 1], 1.0)
    test.assertAlmostEqual(q[0, 0], -0.05, places=5)
    test.assertAlmostEqual(q[1, 0], 0.05, places=5)
    test.assertAlmostEqual(q[0, 1], q[1, 1], places=5)
    test.assertAlmostEqual(qd[0, 1], qd[1, 1], places=5)
    np.testing.assert_array_equal(neighbor_count, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(density[0], density[1], rtol=1e-5, atol=1e-5)
    test.assertGreater(float(density[0]), 0.0)
    np.testing.assert_allclose(state_0.ipbf.constraint.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        state_0.ipbf.constraint_gradient.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(state_0.ipbf.force.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_ipbf_fluid_particle_range_preserves_nonfluid_particles(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 2.0, 3.0),
        ],
        mass=[1.0, 1.0, 5.0],
        radius=[0.05, 0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))

    support_radius = 0.2
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1.0,
            smoothing_radius=support_radius,
            iterations=0,
            fluid_particle_start=0,
            fluid_particle_count=2,
        ),
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()

    dt = 0.05
    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

    q = state_1.particle_q.numpy()
    qd = state_1.particle_qd.numpy()
    density = state_1.ipbf.density.numpy()
    neighbor_count = state_1.ipbf.neighbor_count.numpy()

    expected_fluid_y = 1.0 - 9.81 * dt * dt
    expected_fluid_vy = -9.81 * dt
    expected_fluid_density = kernel_density_contribution(1.0, support_radius, 0.0) + kernel_density_contribution(
        1.0, support_radius, 0.1
    )

    np.testing.assert_allclose(q[:2, 1], np.full(2, expected_fluid_y, dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(qd[:2, 1], np.full(2, expected_fluid_vy, dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(q[2], np.array([0.0, 1.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(qd[2], np.array([1.0, 2.0, 3.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(density[:2], np.full(2, expected_fluid_density, dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(density[2], 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(neighbor_count, np.array([1, 1, 0], dtype=np.int32))


def test_ipbf_boundary_model_contributes_density_and_gradient(test, device):
    state_without_boundary, solver_without_boundary = run_boundary_density_step(device, use_boundary_model=False)
    state_with_boundary, solver_with_boundary = run_boundary_density_step(device, use_boundary_model=True)

    density_without_boundary = state_without_boundary.ipbf.density.numpy()
    density_with_boundary = state_with_boundary.ipbf.density.numpy()
    gradient_without_boundary = state_without_boundary.ipbf.constraint_gradient.numpy()
    gradient_with_boundary = state_with_boundary.ipbf.constraint_gradient.numpy()
    boundary_density = solver_with_boundary._boundary_density.numpy()
    boundary_neighbor_count = solver_with_boundary._boundary_neighbor_count.numpy()

    test.assertIsNone(solver_without_boundary.boundary_model)
    test.assertIsNotNone(solver_with_boundary.boundary_model)
    test.assertTrue(np.all(boundary_density > 0.0))
    test.assertTrue(np.all(boundary_neighbor_count > 0))
    test.assertTrue(np.all(density_with_boundary > density_without_boundary))
    test.assertTrue(np.all(np.linalg.norm(gradient_with_boundary - gradient_without_boundary, axis=1) > 0.0))
    test.assertTrue(
        np.all(state_with_boundary.ipbf.neighbor_count.numpy() > state_without_boundary.ipbf.neighbor_count.numpy())
    )


def test_ipbf_triangle_boundary_samples_contribute_density_and_gradient(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.1),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    rest_density = 1000.0
    support_radius = 0.5
    boundary_thickness = 0.2
    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=support_radius,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=boundary_thickness,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            iterations=0,
            use_constraint_clamp=False,
            fluid_particle_start=0,
            fluid_particle_count=1,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    density = state_1.ipbf.density.numpy()
    constraint_gradient = state_1.ipbf.constraint_gradient.numpy()
    boundary_density = solver._boundary_density.numpy()
    boundary_neighbor_count = solver._boundary_neighbor_count.numpy()
    boundary_sample_x = boundary_model.sample_x_world.numpy()[0]
    boundary_sample_v = boundary_model.sample_v_world.numpy()[0]
    sample_volume = boundary_model.sample_volume_hydrostatic.numpy()[0]

    fluid_x = state_1.particle_q.numpy()[0]
    displacement = fluid_x - boundary_sample_x
    expected_self_density = kernel_density_contribution(1.0, support_radius, 0.0)
    expected_boundary_density = rest_density * kernel_density_contribution(sample_volume, support_radius, 0.1)
    expected_gradient = kernel_gradient_contribution(sample_volume, support_radius, displacement)
    expected_boundary_velocity = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0

    np.testing.assert_allclose(boundary_sample_x, [1.0 / 3.0, 1.0 / 3.0, 0.0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary_sample_v, expected_boundary_velocity, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(sample_volume, 0.5 * boundary_thickness, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(boundary_density[0], expected_boundary_density, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(density[0], expected_self_density + expected_boundary_density, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(constraint_gradient[0], expected_gradient, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_array_equal(boundary_neighbor_count, np.array([1, 0, 0, 0], dtype=np.int32))
    np.testing.assert_allclose(density[1:], np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(constraint_gradient[1:], np.zeros((3, 3), dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_triangle_boundary_pressure_reaction_scatters_to_vertices(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.1),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=1,
            use_constraint_clamp=False,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_pressure_reaction_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    sample_force = boundary_model.sample_force.numpy()[0]
    vertex_force = boundary_model.vertex_force.numpy()
    barycentric = boundary_model.sample_barycentric.numpy()[0]

    test.assertGreater(float(np.linalg.norm(sample_force)), 0.0)
    np.testing.assert_allclose(barycentric, np.full(3, 1.0 / 3.0, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[0], np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[1], barycentric[0] * sample_force, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(vertex_force[2], barycentric[1] * sample_force, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(vertex_force[3], barycentric[2] * sample_force, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(np.sum(vertex_force, axis=0), sample_force, rtol=1.0e-5, atol=1.0e-5)

    boundary_model.clear_forces()
    np.testing.assert_allclose(
        boundary_model.vertex_force.numpy(), np.zeros_like(vertex_force), rtol=1.0e-6, atol=1.0e-6
    )


def test_ipbf_triangle_contact_projects_fluid_and_records_vertex_delta(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=0,
            use_constraint_clamp=False,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    initial_q = state_0.particle_q.numpy().copy()

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    q = state_1.particle_q.numpy()
    vertex_delta = boundary_model.vertex_contact_delta.numpy()
    vertex_force = boundary_model.vertex_force.numpy()
    barycentric = np.full(3, 1.0 / 3.0, dtype=np.float32)
    expected_fluid_delta = np.array([0.0, 0.0, 0.0225], dtype=np.float32)
    expected_vertex_delta = np.array([0.0, 0.0, -0.0075], dtype=np.float32)
    expected_vertex_force = expected_vertex_delta / (0.05 * 0.05)

    test.assertEqual(boundary_model.triangle_count, 1)
    test.assertTrue(solver.fsi_triangle_contact_use_bvh)
    test.assertTrue(solver.fsi_triangle_contact_use_grid)
    test.assertGreaterEqual(solver.fsi_triangle_contact_pair_capacity, 1)
    test.assertIsNotNone(boundary_model.triangle_contact_bvh)
    test.assertIsNotNone(boundary_model.triangle_contact_grid)
    test.assertGreaterEqual(int(solver._triangle_contact_pair_count.numpy()[0]), 1)
    test.assertEqual(int(solver._triangle_contact_pair_overflow.numpy()[0]), 0)
    np.testing.assert_array_equal(boundary_model.triangle_indices.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_allclose(q[0] - initial_q[0], expected_fluid_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(q[1:], initial_q[1:], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vertex_delta[0], np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vertex_delta[1], expected_vertex_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_delta[2], expected_vertex_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_delta[3], expected_vertex_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_delta[1] + vertex_delta[2] + vertex_delta[3], -expected_fluid_delta)
    np.testing.assert_allclose(vertex_force[0], np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[1], expected_vertex_force, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[2], expected_vertex_force, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(vertex_force[3], expected_vertex_force, rtol=1.0e-5, atol=1.0e-6)

    effective_triangle_point = (
        barycentric[0] * (q[1] + vertex_delta[1])
        + barycentric[1] * (q[2] + vertex_delta[2])
        + barycentric[2] * (q[3] + vertex_delta[3])
    )
    test.assertAlmostEqual(float(q[0, 2] - effective_triangle_point[2]), 0.05, places=5)

    boundary_model.clear_forces()
    np.testing.assert_allclose(
        boundary_model.vertex_contact_delta.numpy(), np.zeros_like(vertex_delta), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        boundary_model.vertex_force.numpy(), np.zeros_like(vertex_force), rtol=1.0e-6, atol=1.0e-6
    )

    solver.fsi_triangle_contact_enabled = False
    state_disabled = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_disabled, control=None, contacts=None, dt=0.05)

    np.testing.assert_allclose(state_disabled.particle_q.numpy(), initial_q, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        boundary_model.vertex_contact_delta.numpy(), np.zeros_like(vertex_delta), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        boundary_model.vertex_force.numpy(), np.zeros_like(vertex_force), rtol=1.0e-6, atol=1.0e-6
    )


def test_ipbf_triangle_contact_grid_matches_brute_force_scan(test, device):
    def run_step(*, use_bvh: bool, use_grid: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            vel=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            ],
            mass=[1.0, 1.0, 1.0, 1.0],
            radius=[0.05, 0.05, 0.05, 0.05],
        )
        builder.add_triangle(1, 2, 3)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )
        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_use_bvh=use_bvh,
                fsi_triangle_contact_use_grid=use_grid,
                fsi_triangle_contact_search_radius=1.0,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            solver._triangle_contact_particle_delta.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            boundary_model.vertex_force.numpy(),
        )

    grid_q, grid_particle_delta, grid_vertex_delta, grid_vertex_force = run_step(use_bvh=False, use_grid=True)
    scan_q, scan_particle_delta, scan_vertex_delta, scan_vertex_force = run_step(use_bvh=False, use_grid=False)

    np.testing.assert_allclose(grid_q, scan_q, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(grid_particle_delta, scan_particle_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(grid_vertex_delta, scan_vertex_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(grid_vertex_force, scan_vertex_force, rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_triangle_contact_bvh_matches_brute_force_scan(test, device):
    def run_step(use_bvh: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            vel=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            ],
            mass=[1.0, 1.0, 1.0, 1.0],
            radius=[0.05, 0.05, 0.05, 0.05],
        )
        builder.add_triangle(1, 2, 3)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )
        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_use_bvh=use_bvh,
                fsi_triangle_contact_use_grid=False,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            solver._triangle_contact_particle_delta.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            boundary_model.vertex_force.numpy(),
        )

    bvh_q, bvh_particle_delta, bvh_vertex_delta, bvh_vertex_force = run_step(use_bvh=True)
    scan_q, scan_particle_delta, scan_vertex_delta, scan_vertex_force = run_step(use_bvh=False)

    np.testing.assert_allclose(bvh_q, scan_q, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(bvh_particle_delta, scan_particle_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(bvh_vertex_delta, scan_vertex_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(bvh_vertex_force, scan_vertex_force, rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_triangle_contact_bvh_matches_scan_for_coplanar_patch(test, device):
    def run_step(use_bvh: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0, 1.0, 0.02),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(2.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
                wp.vec3(1.0, 1.0, 0.0),
                wp.vec3(2.0, 1.0, 0.0),
                wp.vec3(0.0, 2.0, 0.0),
                wp.vec3(1.0, 2.0, 0.0),
                wp.vec3(2.0, 2.0, 0.0),
            ],
            vel=[wp.vec3(0.0, 0.0, 0.0)] * 10,
            mass=[1.0] * 10,
            radius=[0.05] * 10,
        )
        builder.add_triangle(1, 2, 5)
        builder.add_triangle(1, 5, 4)
        builder.add_triangle(2, 3, 6)
        builder.add_triangle(2, 6, 5)
        builder.add_triangle(4, 5, 8)
        builder.add_triangle(4, 8, 7)
        builder.add_triangle(5, 6, 9)
        builder.add_triangle(5, 9, 8)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )
        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_use_bvh=use_bvh,
                fsi_triangle_contact_use_grid=False,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            int(solver._triangle_contact_pair_count.numpy()[0]),
        )

    bvh_q, bvh_vertex_delta, bvh_pair_count = run_step(use_bvh=True)
    scan_q, scan_vertex_delta, _scan_pair_count = run_step(use_bvh=False)

    test.assertGreaterEqual(bvh_pair_count, 1)
    np.testing.assert_allclose(bvh_q, scan_q, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(bvh_vertex_delta, scan_vertex_delta, rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_triangle_contact_bvh_pair_overflow_falls_back_to_scan(test, device):
    def run_step(*, use_bvh: bool, pair_capacity: int | None):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
                wp.vec3(0.0, 0.0, 0.01),
                wp.vec3(1.0, 0.0, 0.01),
                wp.vec3(0.0, 1.0, 0.01),
            ],
            vel=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            ],
            mass=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            radius=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        )
        builder.add_triangle(1, 2, 3)
        builder.add_triangle(4, 5, 6)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )
        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_use_bvh=use_bvh,
                fsi_triangle_contact_use_grid=False,
                fsi_triangle_contact_pair_capacity=pair_capacity,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            solver._triangle_contact_particle_delta.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            boundary_model.vertex_force.numpy(),
            int(solver._triangle_contact_pair_count.numpy()[0]),
            int(solver._triangle_contact_pair_overflow.numpy()[0]),
        )

    overflow_q, overflow_particle_delta, overflow_vertex_delta, overflow_vertex_force, overflow_pair_count, overflow_flag = run_step(
        use_bvh=True,
        pair_capacity=1,
    )
    scan_q, scan_particle_delta, scan_vertex_delta, scan_vertex_force, _scan_pair_count, scan_overflow_flag = run_step(
        use_bvh=False,
        pair_capacity=None,
    )

    test.assertGreater(overflow_pair_count, 1)
    test.assertEqual(overflow_flag, 1)
    test.assertEqual(scan_overflow_flag, 0)
    np.testing.assert_allclose(overflow_q, scan_q, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(overflow_particle_delta, scan_particle_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(overflow_vertex_delta, scan_vertex_delta, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(overflow_vertex_force, scan_vertex_force, rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_triangle_contact_pair_cache_reuses_pairs_within_skin(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=2,
            relaxation=1.0,
            use_constraint_clamp=False,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_triangle_contact_use_bvh=True,
            fsi_triangle_contact_use_grid=False,
            fsi_triangle_contact_pair_cache_skin=0.2,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    test.assertEqual(int(solver._triangle_contact_pair_overflow.numpy()[0]), 0)
    test.assertEqual(int(solver._triangle_contact_pair_cache_valid.numpy()[0]), 1)
    test.assertEqual(int(solver._triangle_contact_pair_cache_reuse.numpy()[0]), 1)
    test.assertGreaterEqual(int(solver._triangle_contact_pair_count.numpy()[0]), 1)
    test.assertGreaterEqual(float(solver._triangle_contact_pair_cache_displacement_max.numpy()[0]), 0.0)


def test_ipbf_triangle_contact_pair_cache_ignores_unpaired_fluid_motion(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.02),
            wp.vec3(4.0, 4.0, 4.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(2, 3, 4)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=1,
            relaxation=1.0,
            use_constraint_clamp=False,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=2,
            fsi_triangle_contact_use_bvh=True,
            fsi_triangle_contact_use_grid=False,
            fsi_triangle_contact_pair_cache_skin=0.2,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    first_pair_count = int(solver._triangle_contact_pair_count.numpy()[0])
    first_pair_particles = solver._triangle_contact_pair_particle.numpy()[:first_pair_count]

    test.assertEqual(int(solver._triangle_contact_pair_cache_valid.numpy()[0]), 1)
    test.assertGreaterEqual(first_pair_count, 1)
    test.assertIn(0, first_pair_particles.tolist())
    test.assertNotIn(1, first_pair_particles.tolist())

    moved_q = state_1.particle_q.numpy()
    moved_q[1] = moved_q[1] + np.array([0.18, -0.16, 0.14], dtype=np.float32)
    state_1.particle_q.assign(moved_q)
    state_1.particle_qd.zero_()
    state_1.clear_forces()

    state_2 = model.state()
    solver.step(state_1, state_2, control=None, contacts=None, dt=0.05)

    second_pair_count = int(solver._triangle_contact_pair_count.numpy()[0])
    second_pair_particles = solver._triangle_contact_pair_particle.numpy()[:second_pair_count]
    far_motion = float(np.linalg.norm(np.array([0.18, -0.16, 0.14], dtype=np.float32)))

    test.assertEqual(int(solver._triangle_contact_pair_overflow.numpy()[0]), 0)
    test.assertEqual(int(solver._triangle_contact_pair_cache_valid.numpy()[0]), 1)
    test.assertEqual(int(solver._triangle_contact_pair_cache_reuse.numpy()[0]), 1)
    test.assertGreaterEqual(second_pair_count, 1)
    test.assertIn(0, second_pair_particles.tolist())
    test.assertNotIn(1, second_pair_particles.tolist())
    test.assertLess(float(solver._triangle_contact_pair_cache_displacement_max.numpy()[0]), far_motion - 1.0e-2)


def test_ipbf_triangle_contact_pair_cache_does_not_reuse_empty_pair_set(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.70),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        vel=[wp.vec3(0.0, 0.0, 0.0)] * 4,
        mass=[1.0, 1.0, 1.0, 1.0],
        radius=[0.05, 0.05, 0.05, 0.05],
    )
    builder.add_triangle(1, 2, 3)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    boundary_model = FSIBoundaryModel(
        model,
        spacing=2.0,
        support_radius=0.5,
        include_static=False,
        include_dynamic=False,
        include_triangles=True,
        deformable_sample_thickness=0.2,
        device=device,
    )
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=1000.0,
            smoothing_radius=0.5,
            iterations=1,
            relaxation=1.0,
            use_constraint_clamp=False,
            damping_beta=0.0,
            fluid_particle_start=0,
            fluid_particle_count=1,
            fsi_triangle_contact_use_bvh=True,
            fsi_triangle_contact_use_grid=False,
            fsi_triangle_contact_pair_cache_skin=0.2,
            fsi_triangle_contact_relaxation=1.0,
        ),
        boundary_model=boundary_model,
    )

    state_0 = model.state()
    state_1 = model.state()
    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    test.assertEqual(int(solver._triangle_contact_pair_count.numpy()[0]), 0)
    test.assertEqual(int(solver._triangle_contact_pair_cache_reuse.numpy()[0]), 0)
    test.assertEqual(int(solver._triangle_contact_pair_cache_valid.numpy()[0]), 1)

    moved_q = state_1.particle_q.numpy()
    moved_q[0] = np.array([1.0 / 3.0, 1.0 / 3.0, 0.02], dtype=np.float32)
    state_1.particle_q.assign(moved_q)
    state_1.particle_qd.zero_()
    state_1.clear_forces()

    state_2 = model.state()
    solver.step(state_1, state_2, control=None, contacts=None, dt=0.05)

    test.assertGreaterEqual(int(solver._triangle_contact_pair_count.numpy()[0]), 1)
    test.assertEqual(int(solver._triangle_contact_pair_cache_reuse.numpy()[0]), 0)


def test_ipbf_triangle_contact_prevents_swept_side_change(test, device):
    def run_step(continuous_enabled: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.1),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            vel=[
                wp.vec3(0.0, 0.0, -4.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            ],
            mass=[1.0, 1.0, 1.0, 1.0],
            radius=[0.05, 0.05, 0.05, 0.05],
        )
        builder.add_triangle(1, 2, 3)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )
        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_continuous_enabled=continuous_enabled,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            solver._triangle_contact_particle_delta.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            boundary_model.vertex_force.numpy(),
        )

    enabled_q, enabled_particle_delta, enabled_vertex_delta, enabled_vertex_force = run_step(continuous_enabled=True)
    disabled_q, disabled_particle_delta, disabled_vertex_delta, disabled_vertex_force = run_step(
        continuous_enabled=False
    )

    expected_fluid_delta = np.array([0.0, 0.0, 0.1125], dtype=np.float32)
    expected_vertex_delta = np.array([0.0, 0.0, -0.0375], dtype=np.float32)
    expected_vertex_force = expected_vertex_delta / (0.05 * 0.05)
    expected_vertex_delta_stack = np.repeat(expected_vertex_delta.reshape(1, 3), 3, axis=0)
    expected_vertex_force_stack = np.repeat(expected_vertex_force.reshape(1, 3), 3, axis=0)

    test.assertGreater(float(enabled_q[0, 2]), 0.0)
    np.testing.assert_allclose(enabled_q[0], [1.0 / 3.0, 1.0 / 3.0, 0.0125], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_particle_delta[0], expected_fluid_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_vertex_delta[1:], expected_vertex_delta_stack, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_vertex_force[1:], expected_vertex_force_stack, rtol=1.0e-5, atol=1.0e-6)
    effective_triangle_point = np.array([1.0 / 3.0, 1.0 / 3.0, -0.0375], dtype=np.float32)
    test.assertAlmostEqual(float(enabled_q[0, 2] - effective_triangle_point[2]), 0.05, places=5)

    test.assertLess(float(disabled_q[0, 2]), 0.0)
    np.testing.assert_allclose(disabled_particle_delta, np.zeros_like(disabled_particle_delta), atol=1.0e-6)
    np.testing.assert_allclose(disabled_vertex_delta, np.zeros_like(disabled_vertex_delta), atol=1.0e-6)
    np.testing.assert_allclose(disabled_vertex_force, np.zeros_like(disabled_vertex_force), atol=1.0e-6)


def test_ipbf_triangle_contact_detects_moving_triangle_sweep(test, device):
    def run_step(continuous_enabled: bool):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverIPBF.register_custom_attributes(builder)

        builder.add_particles(
            pos=[
                wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.1),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 1.0, 0.0),
            ],
            vel=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            ],
            mass=[1.0, 1.0, 1.0, 1.0],
            radius=[0.05, 0.05, 0.05, 0.05],
        )
        builder.add_triangle(1, 2, 3)
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))

        boundary_model = FSIBoundaryModel(
            model,
            spacing=2.0,
            support_radius=0.5,
            include_static=False,
            include_dynamic=False,
            include_triangles=True,
            deformable_sample_thickness=0.2,
            device=device,
        )

        history_state = model.state()
        boundary_model.initialize_deformable_contact_history(history_state)

        solver = SolverIPBF(
            model,
            SolverIPBF.Config(
                rest_density=1000.0,
                smoothing_radius=0.5,
                iterations=0,
                use_constraint_clamp=False,
                damping_beta=0.0,
                fluid_particle_start=0,
                fluid_particle_count=1,
                fsi_triangle_contact_continuous_enabled=continuous_enabled,
                fsi_triangle_contact_relaxation=1.0,
            ),
            boundary_model=boundary_model,
        )

        state_0 = model.state()
        state_1 = model.state()
        moved_q = state_0.particle_q.numpy()
        moved_q[1:, 2] = 0.2
        state_0.particle_q.assign(moved_q)
        state_0.particle_qd.zero_()
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

        return (
            state_1.particle_q.numpy(),
            solver._triangle_contact_particle_delta.numpy(),
            boundary_model.vertex_contact_delta.numpy(),
            boundary_model.vertex_force.numpy(),
        )

    enabled_q, enabled_particle_delta, enabled_vertex_delta, enabled_vertex_force = run_step(continuous_enabled=True)
    disabled_q, disabled_particle_delta, disabled_vertex_delta, disabled_vertex_force = run_step(
        continuous_enabled=False
    )

    expected_fluid_delta = np.array([0.0, 0.0, 0.1125], dtype=np.float32)
    expected_vertex_delta = np.array([0.0, 0.0, -0.0375], dtype=np.float32)
    expected_vertex_force = expected_vertex_delta / (0.05 * 0.05)
    expected_vertex_delta_stack = np.repeat(expected_vertex_delta.reshape(1, 3), 3, axis=0)
    expected_vertex_force_stack = np.repeat(expected_vertex_force.reshape(1, 3), 3, axis=0)

    np.testing.assert_allclose(enabled_q[0], [1.0 / 3.0, 1.0 / 3.0, 0.2125], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_particle_delta[0], expected_fluid_delta, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_vertex_delta[1:], expected_vertex_delta_stack, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(enabled_vertex_force[1:], expected_vertex_force_stack, rtol=1.0e-5, atol=1.0e-6)

    np.testing.assert_allclose(disabled_q[0], [1.0 / 3.0, 1.0 / 3.0, 0.1], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(disabled_particle_delta, np.zeros_like(disabled_particle_delta), atol=1.0e-6)
    np.testing.assert_allclose(disabled_vertex_delta, np.zeros_like(disabled_vertex_delta), atol=1.0e-6)
    np.testing.assert_allclose(disabled_vertex_force, np.zeros_like(disabled_vertex_force), atol=1.0e-6)


def test_ipbf_boundary_model_density_stays_consistent_across_sample_spacing(test, device):
    _, coarse_solver = run_boundary_density_step(device, use_boundary_model=True, boundary_spacing=0.25)
    _, fine_solver = run_boundary_density_step(device, use_boundary_model=True, boundary_spacing=0.125)

    coarse_boundary_density = coarse_solver._boundary_density.numpy()
    fine_boundary_density = fine_solver._boundary_density.numpy()

    test.assertTrue(np.all(coarse_boundary_density > 0.0))
    test.assertTrue(np.all(fine_boundary_density > 0.0))
    np.testing.assert_allclose(coarse_boundary_density, fine_boundary_density, rtol=0.15, atol=1.0e-3)


def test_ipbf_static_boundary_weight_scales_static_density_contributions(test, device):
    _, full_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="static",
        static_boundary_weight=1.0,
    )
    _, half_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="static",
        static_boundary_weight=0.5,
    )
    _, zero_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="static",
        static_boundary_weight=0.0,
    )

    full_density = full_solver._boundary_density.numpy()
    half_density = half_solver._boundary_density.numpy()
    zero_density = zero_solver._boundary_density.numpy()

    test.assertTrue(np.all(full_density > 0.0))
    np.testing.assert_allclose(half_density, 0.5 * full_density, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(zero_density, np.zeros_like(full_density), rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_static_boundary_weight_does_not_change_dynamic_density_contributions(test, device):
    _, full_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="dynamic",
        static_boundary_weight=1.0,
    )
    _, zero_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="dynamic",
        static_boundary_weight=0.0,
    )

    np.testing.assert_allclose(
        zero_solver._boundary_density.numpy(),
        full_solver._boundary_density.numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_ipbf_dynamic_boundary_hydrostatic_volume_mode_scales_density_contributions(test, device):
    _, raw_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="dynamic",
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.NONE,
    )
    _, normalized_solver = run_boundary_density_step(
        device,
        use_boundary_model=True,
        boundary_body="dynamic",
        hydrostatic_volume_mode=FSIBoundaryModel.HydrostaticVolumeMode.DYNAMIC_BOX_VOLUME,
    )

    raw_boundary_density = raw_solver._boundary_density.numpy()
    normalized_boundary_density = normalized_solver._boundary_density.numpy()
    raw_sample_volume_sum = float(np.sum(raw_solver.boundary_model.sample_volume.numpy()))
    normalized_sample_volume_sum = float(np.sum(normalized_solver.boundary_model.sample_volume_hydrostatic.numpy()))
    expected_scale = normalized_sample_volume_sum / raw_sample_volume_sum

    test.assertTrue(np.all(raw_boundary_density > 0.0))
    test.assertTrue(np.all(normalized_boundary_density > 0.0))
    test.assertLess(expected_scale, 1.0)
    np.testing.assert_allclose(
        normalized_boundary_density,
        expected_scale * raw_boundary_density,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_ipbf_boundary_pressure_reaction_accumulates_body_wrench(test, device):
    state_0, state_1, solver, boundary_model, body = run_dynamic_box_pressure_reaction_step(
        device, reaction_relaxation=1.0
    )
    _, _, _, disabled_boundary_model, disabled_body = run_dynamic_box_pressure_reaction_step(
        device, reaction_relaxation=0.0
    )

    body_force = boundary_model.body_force.numpy()[body]
    sample_force_norm = np.linalg.norm(boundary_model.sample_force.numpy(), axis=1)
    disabled_body_force = disabled_boundary_model.body_force.numpy()[disabled_body]
    disabled_sample_force = disabled_boundary_model.sample_force.numpy()

    test.assertGreater(float(solver._boundary_density.numpy()[0]), 0.0)
    test.assertGreater(float(state_1.particle_q.numpy()[0, 1]), float(state_0.particle_q.numpy()[0, 1]))
    test.assertGreater(float(np.max(sample_force_norm)), 0.0)
    test.assertLess(float(body_force[1]), 0.0)
    test.assertLess(abs(float(body_force[0])), 1.0e-3)
    test.assertLess(abs(float(body_force[2])), 1.0e-3)
    np.testing.assert_allclose(disabled_body_force, np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(disabled_sample_force, np.zeros_like(disabled_sample_force), rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_computes_constraint_and_gradient(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.0, 0.0),
        ],
        vel=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        mass=[1.0, 1.0],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    rest_density = 1.0
    support_radius = 0.2
    regularization = 1.0e-6
    solver = SolverIPBF(
        model,
        SolverIPBF.Config(
            rest_density=rest_density,
            smoothing_radius=support_radius,
            hessian_regularization=regularization,
            iterations=0,
            use_constraint_clamp=False,
        ),
    )

    state_0 = model.state()
    state_1 = model.state()

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    density = state_1.ipbf.density.numpy()
    constraint = state_1.ipbf.constraint.numpy()
    constraint_gradient = state_1.ipbf.constraint_gradient.numpy()
    force = state_1.ipbf.force.numpy()
    hessian = state_1.ipbf.hessian.numpy()

    distance = 0.1
    expected_density = kernel_density_contribution(1.0, support_radius, 0.0) + kernel_density_contribution(
        1.0, support_radius, distance
    )
    expected_constraint = expected_density / rest_density - 1.0
    expected_gradient_0 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_gradient_1 = -expected_gradient_0
    expected_constraint_hessian_0 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_constraint_hessian_1 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_neighbor_gradient_0 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_neighbor_gradient_1 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_neighbor_constraint_hessian_0 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_neighbor_constraint_hessian_1 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_0 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_0,
        expected_constraint,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_1 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_1,
        expected_constraint,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_hessian_0 = expected_ipbf_hessian(
        expected_gradient_0,
        mass=1.0,
        compliance=0.0,
        dt=0.05,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_0,
        neighbor_gradients=[expected_neighbor_gradient_0],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_0],
    )
    expected_hessian_1 = expected_ipbf_hessian(
        expected_gradient_1,
        mass=1.0,
        compliance=0.0,
        dt=0.05,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_1,
        neighbor_gradients=[expected_neighbor_gradient_1],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_1],
    )

    np.testing.assert_allclose(
        density, np.array([expected_density, expected_density], dtype=np.float32), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        constraint,
        np.array([expected_constraint, expected_constraint], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(constraint_gradient[0], expected_gradient_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(constraint_gradient[1], expected_gradient_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force[0], expected_force_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force[1], expected_force_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hessian[0], expected_hessian_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hessian[1], expected_hessian_1, rtol=1e-5, atol=1e-5)
    test.assertGreater(float(constraint[0]), 0.0)


def test_ipbf_single_iteration_applies_relaxed_jacobi_update(test, device):
    rest_density = 1.0
    support_radius = 0.2
    relaxation = 0.5
    regularization = 1.0e-6
    dt = 0.05
    state_1 = run_two_particle_ipbf_step(
        device,
        iterations=1,
        relaxation=relaxation,
        rest_density=rest_density,
        support_radius=support_radius,
        regularization=regularization,
    )

    distance = 0.1
    expected_density = kernel_density_contribution(1.0, support_radius, 0.0) + kernel_density_contribution(
        1.0, support_radius, distance
    )
    expected_constraint = expected_density / rest_density - 1.0
    expected_gradient_0 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_gradient_1 = -expected_gradient_0
    expected_constraint_hessian_0 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_constraint_hessian_1 = expected_constraint_hessian(
        [1.0, 1.0],
        support_radius,
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        ],
        rest_density,
    )
    expected_neighbor_gradient_0 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_neighbor_gradient_1 = (
        kernel_gradient_contribution(
            1.0,
            support_radius,
            np.array([-distance, 0.0, 0.0], dtype=np.float32),
        )
        / rest_density
    )
    expected_neighbor_constraint_hessian_0 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_neighbor_constraint_hessian_1 = expected_neighbor_constraint_hessian(
        1.0,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_0 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_0,
        expected_constraint,
        support_radius,
        np.array([distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_force_1 = expected_two_particle_force(
        expected_constraint,
        expected_gradient_1,
        expected_constraint,
        support_radius,
        np.array([-distance, 0.0, 0.0], dtype=np.float32),
        rest_density,
    )
    expected_hessian_0 = expected_ipbf_hessian(
        expected_gradient_0,
        mass=1.0,
        compliance=0.0,
        dt=dt,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_0,
        neighbor_gradients=[expected_neighbor_gradient_0],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_0],
    )
    expected_hessian_1 = expected_ipbf_hessian(
        expected_gradient_1,
        mass=1.0,
        compliance=0.0,
        dt=dt,
        regularization=regularization,
        constraint=expected_constraint,
        constraint_hessian=expected_constraint_hessian_1,
        neighbor_gradients=[expected_neighbor_gradient_1],
        neighbor_constraints=[expected_constraint],
        neighbor_constraint_hessians=[expected_neighbor_constraint_hessian_1],
    )
    expected_delta_0 = expected_ipbf_delta(expected_force_0, expected_hessian_0)
    expected_delta_1 = expected_ipbf_delta(expected_force_1, expected_hessian_1)
    expected_q_0 = np.array([-0.05, 1.0, 0.0], dtype=np.float32) + relaxation * expected_delta_0
    expected_q_1 = np.array([0.05, 1.0, 0.0], dtype=np.float32) + relaxation * expected_delta_1
    expected_qd_0 = relaxation * expected_delta_0 / dt
    expected_qd_1 = relaxation * expected_delta_1 / dt

    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy()[0], expected_delta_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy()[1], expected_delta_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[0], expected_q_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_q.numpy()[1], expected_q_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[0], expected_q_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy()[1], expected_q_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_qd_0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_1.particle_qd.numpy()[1], expected_qd_1, rtol=1e-5, atol=1e-5)
    test.assertLess(float(state_1.particle_q.numpy()[0, 0]), -0.05)
    test.assertGreater(float(state_1.particle_q.numpy()[1, 0]), 0.05)


def test_ipbf_kernel_interface_matches_reference(test, device):
    support_radius = 0.3
    displacement = np.array([[0.05, -0.04, 0.02]], dtype=np.float32)

    displacement_wp = wp.array(displacement, dtype=wp.vec3, device=device)
    value_wp = wp.empty(1, dtype=float, device=device)
    gradient_wp = wp.empty(1, dtype=wp.vec3, device=device)
    hessian_wp = wp.empty(1, dtype=wp.mat33, device=device)

    for kernel_family in (KernelFamily.CUBIC_SPLINE, KernelFamily.POLY6):
        wp.launch(
            evaluate_kernel_interface,
            dim=1,
            inputs=[displacement_wp, support_radius, int(kernel_family)],
            outputs=[value_wp, gradient_wp, hessian_wp],
            device=device,
        )

        expected_value = kernel_density_contribution(
            1.0,
            support_radius,
            float(np.linalg.norm(displacement[0])),
            kernel_family,
        )
        expected_gradient = kernel_gradient_contribution(1.0, support_radius, displacement[0], kernel_family)
        expected_hessian = kernel_hessian_reference(support_radius, displacement[0], kernel_family)

        np.testing.assert_allclose(value_wp.numpy()[0], expected_value, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gradient_wp.numpy()[0], expected_gradient, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hessian_wp.numpy()[0], expected_hessian, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hessian_wp.numpy()[0], hessian_wp.numpy()[0].T, rtol=1e-6, atol=1e-6)


def test_ipbf_multiple_iterations_reduce_constraint_magnitude(test, device):
    relaxation = 0.2
    state_0 = run_two_particle_ipbf_step(device, iterations=0, relaxation=relaxation)
    state_1 = run_two_particle_ipbf_step(device, iterations=1, relaxation=relaxation)
    state_2 = run_two_particle_ipbf_step(device, iterations=2, relaxation=relaxation)

    mean_abs_constraint_0 = float(np.mean(np.abs(state_0.ipbf.constraint.numpy())))
    mean_abs_constraint_1 = float(np.mean(np.abs(state_1.ipbf.constraint.numpy())))
    mean_abs_constraint_2 = float(np.mean(np.abs(state_2.ipbf.constraint.numpy())))

    test.assertGreater(mean_abs_constraint_0, mean_abs_constraint_1)
    test.assertGreater(mean_abs_constraint_1, mean_abs_constraint_2)
    test.assertGreater(float(np.abs(state_1.ipbf.delta_q.numpy()).max()), 0.0)
    test.assertGreater(float(np.abs(state_2.ipbf.delta_q.numpy()).max()), 0.0)
    test.assertLess(float(state_1.particle_q.numpy()[0, 0]), -0.05)
    test.assertLess(float(state_2.particle_q.numpy()[0, 0]), float(state_1.particle_q.numpy()[0, 0]))
    test.assertGreater(float(state_1.particle_q.numpy()[1, 0]), 0.05)
    test.assertGreater(float(state_2.particle_q.numpy()[1, 0]), float(state_1.particle_q.numpy()[1, 0]))


def test_ipbf_poly6_step_runs_and_updates_state(test, device):
    state_1 = run_two_particle_ipbf_step(
        device,
        iterations=1,
        relaxation=0.5,
        kernel_family=KernelFamily.POLY6,
    )

    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.array([1, 1], dtype=np.int32))
    test.assertTrue(np.isfinite(state_1.ipbf.density.numpy()).all())
    test.assertTrue(np.isfinite(state_1.ipbf.constraint.numpy()).all())
    test.assertTrue(np.isfinite(state_1.ipbf.force.numpy()).all())
    test.assertTrue(np.isfinite(state_1.ipbf.hessian.numpy()).all())
    test.assertTrue(np.isfinite(state_1.ipbf.delta_q.numpy()).all())
    test.assertGreater(float(np.abs(state_1.ipbf.delta_q.numpy()).max()), 0.0)


def test_ipbf_reset_restores_initial_particle_state(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverIPBF.register_custom_attributes(builder)

    builder.add_particles(
        pos=[
            wp.vec3(-0.05, 1.0, 0.0),
            wp.vec3(0.05, 1.1, 0.0),
        ],
        vel=[
            wp.vec3(0.2, 0.0, 0.0),
            wp.vec3(-0.2, 0.0, 0.0),
        ],
        mass=[1.0, 1.5],
        radius=[0.05, 0.05],
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, -9.81, 0.0))
    solver = SolverIPBF(model, SolverIPBF.Config(smoothing_radius=0.2))

    state_0 = model.state()
    state_1 = model.state()

    q_initial = state_0.particle_q.numpy().copy()
    qd_initial = state_0.particle_qd.numpy().copy()

    state_0.clear_forces()
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.05)

    q_advanced = state_1.particle_q.numpy()
    test.assertFalse(np.allclose(q_advanced, q_initial))

    solver.reset(state_1)

    np.testing.assert_allclose(state_1.particle_q.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.particle_qd.numpy(), qd_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.particle_f.numpy(), np.zeros_like(q_initial), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.y.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.x_guess.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.x_new.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.x_star.numpy(), q_initial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.density.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(state_1.ipbf.neighbor_count.numpy(), np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(state_1.ipbf.constraint.numpy(), np.zeros(2, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        state_1.ipbf.constraint_gradient.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(state_1.ipbf.force.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_1.ipbf.delta_q.numpy(), np.zeros((2, 3), dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_ipbf_artificial_damping_reduces_velocity_magnitude(test, device):
    undamped = run_two_particle_ipbf_step(
        device,
        iterations=1,
        relaxation=0.5,
        damping_beta=0.0,
    )
    damped = run_two_particle_ipbf_step(
        device,
        iterations=1,
        relaxation=0.5,
        damping_compliance=1.0 / 1000.0,
        damping_beta=60.0,
    )

    np.testing.assert_allclose(damped.particle_q.numpy(), undamped.particle_q.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(damped.ipbf.delta_q.numpy(), undamped.ipbf.delta_q.numpy(), rtol=1e-6, atol=1e-6)

    damped_speed = np.linalg.norm(damped.particle_qd.numpy(), axis=1)
    undamped_speed = np.linalg.norm(undamped.particle_qd.numpy(), axis=1)
    test.assertTrue(np.all(damped_speed <= undamped_speed + 1.0e-7))
    test.assertGreater(float(np.max(undamped_speed - damped_speed)), 0.0)
    test.assertTrue(np.all(np.linalg.norm(damped.ipbf.x_star.numpy() - damped.particle_q.numpy(), axis=1) > 0.0))


def test_ipbf_static_shape_boundary_projects_particles_out_of_ground(test, device):
    without_contacts, _ = run_single_particle_ground_step(device, use_contacts=False)
    with_contacts, contacts = run_single_particle_ground_step(device, use_contacts=True)

    test.assertLess(float(without_contacts.particle_q.numpy()[0, 1]), 0.05)
    test.assertGreaterEqual(float(with_contacts.particle_q.numpy()[0, 1]), 0.05 - 1.0e-5)
    test.assertAlmostEqual(float(with_contacts.particle_q.numpy()[0, 0]), 0.0, places=6)
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)


def test_ipbf_projection_reaction_accumulates_body_wrench(test, device):
    state_0, state_1, solver, boundary_model, contacts, body, particle_mass, dt = (
        run_dynamic_box_projection_reaction_step(device)
    )

    total_delta = solver._boundary_projection_delta_total.numpy()[0]
    body_force = boundary_model.body_force.numpy()[body]
    body_torque = boundary_model.body_torque.numpy()[body]
    expected_delta = state_1.particle_q.numpy()[0] - state_0.particle_q.numpy()[0]
    expected_force = -particle_mass * total_delta / (dt * dt)

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    np.testing.assert_allclose(total_delta, expected_delta, rtol=1.0e-5, atol=1.0e-6)
    test.assertGreater(float(total_delta[1]), 0.0)
    np.testing.assert_allclose(body_force, expected_force, rtol=1.0e-5, atol=1.0e-5)
    test.assertLess(float(body_force[1]), 0.0)
    np.testing.assert_allclose(body_torque, np.zeros(3, dtype=np.float32), rtol=1.0e-5, atol=1.0e-5)


def test_ipbf_projection_reaction_uses_dedicated_relaxation(test, device):
    _, _, _, enabled_boundary_model, _, enabled_body, _, _ = run_dynamic_box_projection_reaction_step(
        device,
        projection_reaction_relaxation=1.0,
    )
    _, _, _, disabled_boundary_model, _, disabled_body, _, _ = run_dynamic_box_projection_reaction_step(
        device,
        projection_reaction_relaxation=0.0,
    )

    enabled_force = enabled_boundary_model.body_force.numpy()[enabled_body]
    disabled_force = disabled_boundary_model.body_force.numpy()[disabled_body]

    test.assertGreater(float(np.linalg.norm(enabled_force)), 0.0)
    np.testing.assert_allclose(disabled_force, np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_velocity_projection_reaction_accumulates_body_wrench(test, device):
    state_1, boundary_model, contacts, body = run_dynamic_box_velocity_reaction_step(device)

    body_force = boundary_model.body_force.numpy()[body]

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(float(body_force[0]), 0.0)
    test.assertAlmostEqual(float(body_force[1]), 0.0, places=5)
    test.assertAlmostEqual(float(body_force[2]), 0.0, places=5)
    test.assertLessEqual(float(state_1.particle_qd.numpy()[0, 0]), 1.0e-5)


def test_ipbf_velocity_projection_reaction_uses_dedicated_relaxation(test, device):
    _, enabled_boundary_model, _, enabled_body = run_dynamic_box_velocity_reaction_step(
        device,
        velocity_reaction_relaxation=1.0,
        projection_reaction_relaxation=0.0,
    )
    _, disabled_boundary_model, _, disabled_body = run_dynamic_box_velocity_reaction_step(
        device,
        velocity_reaction_relaxation=0.0,
        projection_reaction_relaxation=0.0,
    )

    enabled_force = enabled_boundary_model.body_force.numpy()[enabled_body]
    disabled_force = disabled_boundary_model.body_force.numpy()[disabled_body]

    test.assertGreater(float(np.linalg.norm(enabled_force)), 0.0)
    np.testing.assert_allclose(disabled_force, np.zeros(3, dtype=np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_legacy_reaction_relaxation_alias_still_drives_contact_reactions(test, device):
    _, _, _, projection_boundary_model, _, projection_body, _, _ = run_dynamic_box_projection_reaction_step(
        device,
        legacy_reaction_relaxation=1.0,
    )
    _, velocity_boundary_model, _, velocity_body = run_dynamic_box_velocity_reaction_step(
        device,
        legacy_reaction_relaxation=1.0,
    )

    test.assertGreater(float(np.linalg.norm(projection_boundary_model.body_force.numpy()[projection_body])), 0.0)
    test.assertGreater(float(np.linalg.norm(velocity_boundary_model.body_force.numpy()[velocity_body])), 0.0)


def test_ipbf_ground_contact_projection_prevents_persistent_downward_velocity(test, device):
    final_state, contacts, positions, velocities = run_single_particle_ground_rollout(device, steps=6)

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreaterEqual(float(np.min(positions[:, 1])), 0.05 - 1.0e-5)
    test.assertGreaterEqual(float(velocities[-1, 1]), -1.0e-5)
    test.assertLessEqual(abs(float(final_state.particle_qd.numpy()[0, 1])), 1.0e-4)


def test_ipbf_box_container_rollout_keeps_particles_inside_bounds(test, device):
    if wp.get_device(device).is_cpu:
        return

    _, max_abs_x, max_abs_z, min_y = run_ipbf_box_container_rollout(device, num_frames=120)

    test.assertLessEqual(max_abs_x, 0.57)
    test.assertLessEqual(max_abs_z, 0.57)
    test.assertGreaterEqual(min_y, -0.02)


def test_ipbf_viscosity_reduces_two_particle_relative_speed(test, device):
    without_viscosity = run_two_particle_ipbf_step(
        device,
        iterations=0,
        relaxation=0.5,
        initial_velocities=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        viscosity_coefficient=0.0,
    )
    with_viscosity = run_two_particle_ipbf_step(
        device,
        iterations=0,
        relaxation=0.5,
        initial_velocities=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        viscosity_coefficient=0.02,
    )

    velocity_difference_without = without_viscosity.particle_qd.numpy()[1] - without_viscosity.particle_qd.numpy()[0]
    velocity_difference_with = with_viscosity.particle_qd.numpy()[1] - with_viscosity.particle_qd.numpy()[0]

    test.assertLess(
        float(np.linalg.norm(velocity_difference_with)),
        float(np.linalg.norm(velocity_difference_without)),
    )


def test_ipbf_boundary_viscosity_reduces_speed_near_static_samples(test, device):
    without_boundary_viscosity = run_static_boundary_viscosity_step(
        device,
        viscosity_boundary_coefficient=0.0,
    )
    with_boundary_viscosity = run_static_boundary_viscosity_step(
        device,
        viscosity_boundary_coefficient=0.02,
    )
    disabled_static_weight = run_static_boundary_viscosity_step(
        device,
        viscosity_boundary_coefficient=0.02,
        static_boundary_weight=0.0,
    )

    speed_without = np.linalg.norm(without_boundary_viscosity.particle_qd.numpy(), axis=1)
    speed_with = np.linalg.norm(with_boundary_viscosity.particle_qd.numpy(), axis=1)
    speed_disabled = np.linalg.norm(disabled_static_weight.particle_qd.numpy(), axis=1)

    test.assertTrue(np.all(speed_with < speed_without))
    np.testing.assert_allclose(speed_disabled, speed_without, rtol=1.0e-6, atol=1.0e-6)


def test_ipbf_ground_contact_tangential_damping_reduces_speed(test, device):
    _, contacts, _, velocities = run_single_particle_ground_rollout(
        device,
        steps=10,
        boundary_velocity_damping=0.9,
        initial_velocity=(1.0, 0.0, 0.0),
        gravity=(0.0, -9.81, 0.0),
        dt=0.05,
    )

    speed_history = np.linalg.norm(velocities[:, [0, 2]], axis=1)

    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertLess(float(speed_history[-1]), float(speed_history[0]))
    test.assertLess(float(speed_history[-1]), 0.5)


devices = get_test_devices(mode="basic")


class TestSolverIPBF(unittest.TestCase):
    pass


add_function_test(
    TestSolverIPBF,
    "test_ipbf_registers_attributes_and_applies_inertial_prediction",
    test_ipbf_registers_attributes_and_applies_inertial_prediction,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_requires_registered_attributes",
    test_ipbf_requires_registered_attributes,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_multi_particle_shell_step_builds_neighbor_search",
    test_ipbf_multi_particle_shell_step_builds_neighbor_search,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_fluid_particle_range_preserves_nonfluid_particles",
    test_ipbf_fluid_particle_range_preserves_nonfluid_particles,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_computes_constraint_and_gradient",
    test_ipbf_computes_constraint_and_gradient,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_single_iteration_applies_relaxed_jacobi_update",
    test_ipbf_single_iteration_applies_relaxed_jacobi_update,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_kernel_interface_matches_reference",
    test_ipbf_kernel_interface_matches_reference,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_multiple_iterations_reduce_constraint_magnitude",
    test_ipbf_multiple_iterations_reduce_constraint_magnitude,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_poly6_step_runs_and_updates_state",
    test_ipbf_poly6_step_runs_and_updates_state,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_artificial_damping_reduces_velocity_magnitude",
    test_ipbf_artificial_damping_reduces_velocity_magnitude,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_reset_restores_initial_particle_state",
    test_ipbf_reset_restores_initial_particle_state,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_static_shape_boundary_projects_particles_out_of_ground",
    test_ipbf_static_shape_boundary_projects_particles_out_of_ground,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_boundary_model_contributes_density_and_gradient",
    test_ipbf_boundary_model_contributes_density_and_gradient,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_boundary_samples_contribute_density_and_gradient",
    test_ipbf_triangle_boundary_samples_contribute_density_and_gradient,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_boundary_pressure_reaction_scatters_to_vertices",
    test_ipbf_triangle_boundary_pressure_reaction_scatters_to_vertices,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_projects_fluid_and_records_vertex_delta",
    test_ipbf_triangle_contact_projects_fluid_and_records_vertex_delta,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_grid_matches_brute_force_scan",
    test_ipbf_triangle_contact_grid_matches_brute_force_scan,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_bvh_matches_brute_force_scan",
    test_ipbf_triangle_contact_bvh_matches_brute_force_scan,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_bvh_matches_scan_for_coplanar_patch",
    test_ipbf_triangle_contact_bvh_matches_scan_for_coplanar_patch,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_bvh_pair_overflow_falls_back_to_scan",
    test_ipbf_triangle_contact_bvh_pair_overflow_falls_back_to_scan,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_pair_cache_reuses_pairs_within_skin",
    test_ipbf_triangle_contact_pair_cache_reuses_pairs_within_skin,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_pair_cache_ignores_unpaired_fluid_motion",
    test_ipbf_triangle_contact_pair_cache_ignores_unpaired_fluid_motion,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_pair_cache_does_not_reuse_empty_pair_set",
    test_ipbf_triangle_contact_pair_cache_does_not_reuse_empty_pair_set,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_prevents_swept_side_change",
    test_ipbf_triangle_contact_prevents_swept_side_change,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_triangle_contact_detects_moving_triangle_sweep",
    test_ipbf_triangle_contact_detects_moving_triangle_sweep,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_boundary_model_density_stays_consistent_across_sample_spacing",
    test_ipbf_boundary_model_density_stays_consistent_across_sample_spacing,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_static_boundary_weight_scales_static_density_contributions",
    test_ipbf_static_boundary_weight_scales_static_density_contributions,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_static_boundary_weight_does_not_change_dynamic_density_contributions",
    test_ipbf_static_boundary_weight_does_not_change_dynamic_density_contributions,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_dynamic_boundary_hydrostatic_volume_mode_scales_density_contributions",
    test_ipbf_dynamic_boundary_hydrostatic_volume_mode_scales_density_contributions,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_boundary_pressure_reaction_accumulates_body_wrench",
    test_ipbf_boundary_pressure_reaction_accumulates_body_wrench,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_projection_reaction_accumulates_body_wrench",
    test_ipbf_projection_reaction_accumulates_body_wrench,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_projection_reaction_uses_dedicated_relaxation",
    test_ipbf_projection_reaction_uses_dedicated_relaxation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_velocity_projection_reaction_accumulates_body_wrench",
    test_ipbf_velocity_projection_reaction_accumulates_body_wrench,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_velocity_projection_reaction_uses_dedicated_relaxation",
    test_ipbf_velocity_projection_reaction_uses_dedicated_relaxation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_legacy_reaction_relaxation_alias_still_drives_contact_reactions",
    test_ipbf_legacy_reaction_relaxation_alias_still_drives_contact_reactions,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_ground_contact_projection_prevents_persistent_downward_velocity",
    test_ipbf_ground_contact_projection_prevents_persistent_downward_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_box_container_rollout_keeps_particles_inside_bounds",
    test_ipbf_box_container_rollout_keeps_particles_inside_bounds,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_viscosity_reduces_two_particle_relative_speed",
    test_ipbf_viscosity_reduces_two_particle_relative_speed,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_boundary_viscosity_reduces_speed_near_static_samples",
    test_ipbf_boundary_viscosity_reduces_speed_near_static_samples,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverIPBF,
    "test_ipbf_ground_contact_tangential_damping_reduces_speed",
    test_ipbf_ground_contact_tangential_damping_reduces_speed,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
