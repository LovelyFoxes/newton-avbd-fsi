"""Import a Newton FSI cache into Blender and render a fixed-frame preview.

Run from Blender, for example:

    blender --background --python .blender/scripts/render_fsi_cache.py -- \
      --cache-dir .blender/cache/fsi_three_sphere_buoys \
      --output-dir .blender/renders/fsi_three_sphere_buoys \
      --frames 30 \
      --render

The script is designed for local thesis/demo rendering, not as Newton runtime
code. It intentionally accepts a flexible cache schema so the Newton exporter can
evolve without rewriting the Blender side immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Directory containing metadata.json and frames/*.npz."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".blender/renders/fsi_preview"))
    parser.add_argument("--config", type=Path, default=Path(".blender/config/three_sphere_buoys_render.json"))
    parser.add_argument("--save-blend", type=Path, default=None, help="Optional .blend output path.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to import/render.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--water-max-particles", type=int, default=None)
    parser.add_argument("--water-radius-scale", type=float, default=None)
    parser.add_argument(
        "--axis-conversion",
        choices=["newton-y-up-to-blender-z-up", "none"],
        default=None,
        help="Coordinate conversion applied to cached Newton simulation data.",
    )
    parser.add_argument("--render", action="store_true", help="Render an image sequence after scene setup.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.2,
    metallic: float = 0.0,
    alpha_blend: bool = False,
    transmission: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    material.blend_method = "BLEND" if alpha_blend else "OPAQUE"
    material.use_screen_refraction = alpha_blend

    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        set_node_input(bsdf, "Base Color", color)
        set_node_input(bsdf, "Alpha", color[3])
        set_node_input(bsdf, "Roughness", roughness)
        set_node_input(bsdf, "Metallic", metallic)
        set_node_input(bsdf, "Transmission Weight", transmission)
        set_node_input(bsdf, "Transmission", transmission)
    return material


def set_node_input(node: bpy.types.Node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_render(config: dict, metadata: dict, output_dir: Path) -> None:
    scene = bpy.context.scene
    resolution = config.get("resolution", [1920, 1080])
    render_config = config.get("render", {})
    export_config = metadata.get("export", {})
    render_fps = export_config.get("actual_video_fps", config.get("fps", 30))

    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.fps = int(round(float(render_fps)))
    scene.render.filepath = str(output_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = bool(render_config.get("transparent_background", False))

    engine = str(render_config.get("engine", "CYCLES")).upper()
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(render_config.get("samples", 96))
        scene.cycles.use_denoising = True
    elif engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        scene.render.engine = engine


def add_camera_and_lights(config: dict) -> None:
    camera_config = config.get("camera", {})
    location = tuple(camera_config.get("location", [1.35, -1.42, 0.92]))
    target = tuple(camera_config.get("target", [0.0, 0.0, 0.42]))

    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "FSI_Camera"
    camera.data.lens = float(camera_config.get("focal_length_mm", 45.0))
    look_at(camera, target)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(0.0, -0.35, 1.9))
    key = bpy.context.object
    key.name = "Softbox_Key"
    key.data.energy = 550.0
    key.data.size = 2.2

    bpy.ops.object.light_add(type="POINT", location=(-0.7, -0.55, 0.65))
    rim = bpy.context.object
    rim.name = "Blue_Rim_Light"
    rim.data.energy = 45.0
    rim.data.color = (0.38, 0.58, 1.0)


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material,
    axis_mode: str,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=to_blender_position(location, axis_mode))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = to_blender_scale(scale, axis_mode)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)


def add_tank(config: dict, metadata: dict, materials: dict[str, bpy.types.Material], axis_mode: str) -> None:
    tank = metadata.get("tank") or config.get("tank", {})
    hx = float(tank.get("half_width", 0.52))
    hz = float(tank.get("half_depth", 0.39))
    hy = float(tank.get("half_height", 0.92))
    wall_t = float(tank.get("wall_thickness", 0.035))
    top_y = 2.0 * hy

    glass = materials["glass"]
    floor = materials["floor"]
    add_box("Tank_Floor", (0.0, -wall_t * 0.5, 0.0), (2.0 * hx, wall_t, 2.0 * hz), floor, axis_mode)
    add_box("Tank_Left_Wall", (-(hx + wall_t * 0.5), hy, 0.0), (wall_t, top_y, 2.0 * hz), glass, axis_mode)
    add_box("Tank_Right_Wall", ((hx + wall_t * 0.5), hy, 0.0), (wall_t, top_y, 2.0 * hz), glass, axis_mode)
    add_box("Tank_Back_Wall", (0.0, hy, -(hz + wall_t * 0.5)), (2.0 * hx, top_y, wall_t), glass, axis_mode)
    add_box("Tank_Front_Wall", (0.0, hy, (hz + wall_t * 0.5)), (2.0 * hx, top_y, wall_t), glass, axis_mode)


def frame_paths(cache_dir: Path | None) -> list[Path]:
    if cache_dir is None:
        return []
    frame_dir = cache_dir / "frames"
    paths = sorted(frame_dir.glob("frame_*.npz")) if frame_dir.exists() else []
    if not paths:
        paths = sorted(cache_dir.glob("frame_*.npz"))
    return paths


def read_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def pick_array(frame: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in frame:
            return np.asarray(frame[name])
    return None


def normalize_quat_xyzw(quat: np.ndarray) -> tuple[float, float, float, float]:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    q = q / norm
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def resolve_axis_mode(args: argparse.Namespace, config: dict, metadata: dict) -> str:
    if args.axis_conversion is not None:
        return args.axis_conversion
    if "axis_conversion" in config:
        return str(config["axis_conversion"])
    coordinate_system = metadata.get("coordinate_system", {})
    return str(coordinate_system.get("blender_recommended_axis_conversion", "newton-y-up-to-blender-z-up"))


def to_blender_position(position: tuple[float, float, float] | np.ndarray, axis_mode: str) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in position)
    if axis_mode == "none":
        return (x, y, z)
    return (x, -z, y)


def to_blender_scale(scale: tuple[float, float, float], axis_mode: str) -> tuple[float, float, float]:
    sx, sy, sz = (float(value) for value in scale)
    if axis_mode == "none":
        return (sx, sy, sz)
    return (sx, sz, sy)


def to_blender_positions(positions: np.ndarray, axis_mode: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if axis_mode == "none":
        return positions
    converted = np.empty_like(positions)
    converted[:, 0] = positions[:, 0]
    converted[:, 1] = -positions[:, 2]
    converted[:, 2] = positions[:, 1]
    return converted


def to_blender_quaternion_wxyz(quat_xyzw: np.ndarray, axis_mode: str) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = normalize_quat_xyzw(quat_xyzw)
    if axis_mode == "none":
        return (qw, qx, qy, qz)

    rotation_newton = mathutils.Quaternion((qw, qx, qy, qz)).to_matrix()
    basis = mathutils.Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    rotation_blender = basis @ rotation_newton @ basis.transposed()
    quat_blender = rotation_blender.to_quaternion()
    return (quat_blender.w, quat_blender.x, quat_blender.y, quat_blender.z)


def sampled_indices(count: int, max_count: int | None) -> np.ndarray:
    if max_count is None or count <= max_count:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, max_count, dtype=np.int32)


def add_preview_water(
    frames: list[dict[str, np.ndarray]],
    config: dict,
    materials: dict[str, bpy.types.Material],
    *,
    max_particles: int | None,
    radius_scale: float | None,
    axis_mode: str,
) -> None:
    if not frames:
        return

    first_positions = pick_array(frames[0], "fluid_positions", "particle_positions", "positions")
    if first_positions is None or len(first_positions) == 0:
        return

    water_config = config.get("water", {})
    max_particles = int(max_particles or water_config.get("max_preview_particles", 2500))
    radius_scale = float(radius_scale or water_config.get("preview_radius_scale", 0.55))
    indices = sampled_indices(len(first_positions), max_particles)

    radii = pick_array(frames[0], "fluid_radii", "particle_radii", "radii")
    base_radius = float(np.median(radii[indices])) if radii is not None and len(radii) else 0.006
    render_radius = base_radius * radius_scale

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=render_radius, location=(0.0, 0.0, 0.0))
    prototype = bpy.context.object
    prototype.name = "Water_Particle_Prototype"
    prototype.data.materials.append(materials["water"])
    prototype.hide_viewport = True
    prototype.hide_render = True

    water_objects: list[bpy.types.Object] = []
    for point_index in indices:
        obj = bpy.data.objects.new(f"water_{int(point_index):05d}", prototype.data)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(materials["water"])
        water_objects.append(obj)

    for frame_number, frame in enumerate(frames, start=1):
        positions = pick_array(frame, "fluid_positions", "particle_positions", "positions")
        if positions is None:
            continue
        positions = to_blender_positions(np.asarray(positions, dtype=np.float32), axis_mode)
        for obj, point_index in zip(water_objects, indices, strict=True):
            if point_index >= len(positions):
                continue
            obj.location = tuple(float(v) for v in positions[point_index])
            obj.keyframe_insert(data_path="location", frame=frame_number)


def body_positions_and_rotations(frame: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    body_q = pick_array(frame, "body_q", "body_transforms")
    if body_q is not None and body_q.ndim == 2 and body_q.shape[1] >= 7:
        return body_q[:, :3], body_q[:, 3:7]
    positions = pick_array(frame, "body_positions", "sphere_positions")
    rotations = pick_array(frame, "body_rotations", "sphere_rotations")
    return positions, rotations


def add_buoys(
    frames: list[dict[str, np.ndarray]],
    metadata: dict,
    config: dict,
    materials: dict[str, bpy.types.Material],
    axis_mode: str,
) -> None:
    if not frames:
        add_template_buoys(config, materials, axis_mode)
        return

    positions, rotations = body_positions_and_rotations(frames[0])
    if positions is None or len(positions) == 0:
        add_template_buoys(config, materials, axis_mode)
        return

    buoy_config = config.get("buoys", {})
    radii = np.asarray(metadata.get("sphere_radii", []), dtype=np.float32)
    if radii.size == 0:
        frame_radii = pick_array(frames[0], "sphere_radii", "body_sphere_radii")
        radii = np.asarray(frame_radii, dtype=np.float32) if frame_radii is not None else np.array([])
    default_radius = float(buoy_config.get("default_radius", 0.07))

    buoy_objects: list[bpy.types.Object] = []
    positions = to_blender_positions(np.asarray(positions, dtype=np.float32), axis_mode)
    for body_index in range(len(positions)):
        radius = float(radii[body_index]) if body_index < len(radii) else default_radius
        material = materials[f"buoy_{body_index % 3}"]
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48, ring_count=24, radius=radius, location=tuple(positions[body_index])
        )
        obj = bpy.context.object
        obj.name = f"Buoy_{body_index:02d}"
        obj.data.materials.append(material)
        buoy_objects.append(obj)

    for frame_number, frame in enumerate(frames, start=1):
        positions, rotations = body_positions_and_rotations(frame)
        if positions is None:
            continue
        positions = to_blender_positions(np.asarray(positions, dtype=np.float32), axis_mode)
        for body_index, obj in enumerate(buoy_objects):
            if body_index >= len(positions):
                continue
            obj.location = tuple(float(v) for v in positions[body_index])
            obj.keyframe_insert(data_path="location", frame=frame_number)
            if rotations is not None and body_index < len(rotations):
                obj.rotation_mode = "QUATERNION"
                obj.rotation_quaternion = to_blender_quaternion_wxyz(rotations[body_index], axis_mode)
                obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)


def add_template_buoys(config: dict, materials: dict[str, bpy.types.Material], axis_mode: str) -> None:
    radius = float(config.get("buoys", {}).get("default_radius", 0.07))
    for index, x in enumerate((-0.24, 0.0, 0.24)):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48, ring_count=24, radius=radius, location=to_blender_position((x, 0.54, 0.0), axis_mode)
        )
        obj = bpy.context.object
        obj.name = f"Template_Buoy_{index}"
        obj.data.materials.append(materials[f"buoy_{index}"])


def build_materials(config: dict) -> dict[str, bpy.types.Material]:
    water_color = tuple(config.get("water", {}).get("material_color", [0.22, 0.58, 1.0, 0.32]))
    buoy_colors = config.get("buoys", {}).get(
        "density_colors",
        [[1.0, 0.62, 0.12, 1.0], [0.28, 0.82, 0.20, 1.0], [0.06, 0.88, 0.86, 1.0]],
    )
    return {
        "water": make_material(
            "Water_Preview", tuple(water_color), roughness=0.03, alpha_blend=True, transmission=0.25
        ),
        "glass": make_material(
            "Tank_Glass", (0.72, 0.85, 1.0, 0.18), roughness=0.02, alpha_blend=True, transmission=0.55
        ),
        "floor": make_material("Tank_Floor_Matte", (0.16, 0.17, 0.18, 1.0), roughness=0.55),
        "buoy_0": make_material("Buoy_Light", tuple(buoy_colors[0]), roughness=0.28),
        "buoy_1": make_material("Buoy_Medium", tuple(buoy_colors[1]), roughness=0.24),
        "buoy_2": make_material("Buoy_Heavy", tuple(buoy_colors[2]), roughness=0.20),
    }


def selected_frames(paths: list[Path], start: int, stride: int, count: int | None) -> list[Path]:
    if not paths:
        return []
    sliced = paths[max(0, start) :: max(1, stride)]
    return sliced if count is None else sliced[: max(0, count)]


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    metadata = load_json(args.cache_dir / "metadata.json") if args.cache_dir is not None else {}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    axis_mode = resolve_axis_mode(args, config, metadata)

    paths = selected_frames(frame_paths(args.cache_dir), args.start_frame, args.stride, args.frames)
    frames = [read_frame(path) for path in paths]
    frame_count = len(frames) if frames else int(args.frames or config.get("frame_count", 30))

    clear_scene()
    configure_render(config, metadata, output_dir)
    materials = build_materials(config)
    add_tank(config, metadata, materials, axis_mode)
    add_camera_and_lights(config)
    add_buoys(frames, metadata, config, materials, axis_mode)
    add_preview_water(
        frames,
        config,
        materials,
        max_particles=args.water_max_particles,
        radius_scale=args.water_radius_scale,
        axis_mode=axis_mode,
    )

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, frame_count)

    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))

    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
