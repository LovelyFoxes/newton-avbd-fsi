from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import render_fsi_cache as cache_preview
import render_fsi_surface_sequence as surface_preview


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Build Blender templates from generic FSI example caches.")
    parser.add_argument("--config", type=Path, default=Path(".blender/config/fsi_experiment_scenes.json"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, default=None)
    parser.add_argument("--save-blend", type=Path, default=None)
    parser.add_argument("--mode", choices=["particles", "surface"], default="particles")
    parser.add_argument("--frame-indices", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_frame(cache_dir: Path, frame_index: int) -> dict[str, np.ndarray]:
    path = cache_dir / "frames" / f"frame_{frame_index:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame cache not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def available_frame_indices(cache_dir: Path) -> list[int]:
    frame_dir = cache_dir / "frames"
    paths = sorted(frame_dir.glob("frame_*.npz"))
    indices: list[int] = []
    for path in paths:
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return indices


def parse_frame_indices(raw: str | None, metadata: dict, cache_dir: Path) -> list[int]:
    if raw and raw.strip().lower() == "auto":
        indices = available_frame_indices(cache_dir)
        if not indices:
            raise FileNotFoundError(f"No cached frames found in {cache_dir / 'frames'}.")
        return indices
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    values = metadata.get("render_frames", [])
    if values:
        return [int(value) for value in values]
    count = int(metadata.get("record_frames", 1))
    return sorted(set([0, max(0, count // 3), max(0, 2 * count // 3), max(0, count - 1)]))


def configure_scene(config: dict, metadata: dict, output_dir: Path) -> None:
    render_cfg = config.get("render", {})
    scene = bpy.context.scene
    resolution = render_cfg.get("resolution", [1920, 1080])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.filepath = str(output_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = int(render_cfg.get("fps", metadata.get("fps", 60)))
    try:
        scene.render.engine = str(render_cfg.get("engine", "BLENDER_EEVEE_NEXT"))
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    if scene.render.engine == "CYCLES":
        configure_cycles_device(render_cfg)
        scene.cycles.samples = int(render_cfg.get("samples", 96))
        scene.cycles.use_denoising = bool(render_cfg.get("denoise", True))
        scene.cycles.max_bounces = int(render_cfg.get("max_bounces", 8))
        scene.cycles.transparent_max_bounces = int(render_cfg.get("transparent_max_bounces", 8))
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(render_cfg.get("samples", 16))
    if hasattr(scene, "eevee_next"):
        scene.eevee_next.taa_render_samples = int(render_cfg.get("samples", 16))
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        cache_preview.set_node_input(bg, "Color", tuple(render_cfg.get("background", [0, 0, 0, 1])))
        cache_preview.set_node_input(bg, "Strength", float(render_cfg.get("background_strength", 1.0)))


def configure_cycles_device(render_cfg: dict) -> None:
    scene = bpy.context.scene
    scene.cycles.device = str(render_cfg.get("device", "GPU")).upper()
    if scene.cycles.device != "GPU":
        return
    preferences = bpy.context.preferences.addons.get("cycles")
    if preferences is None:
        print("[FSI Experiment] Cycles preferences not found; render may fall back to CPU.")
        return
    cycles_preferences = preferences.preferences
    preferred = [str(render_cfg.get("compute_device_type", "OPTIX")).upper(), "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]
    for device_type in dict.fromkeys(preferred):
        try:
            cycles_preferences.compute_device_type = device_type
            cycles_preferences.get_devices()
        except Exception:
            continue
        devices = list(getattr(cycles_preferences, "devices", []))
        gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
        if gpu_devices:
            for device in devices:
                device.use = device in gpu_devices
            print(f"[FSI Experiment] Cycles GPU device type: {device_type}")
            return
    print("[FSI Experiment] No Cycles GPU device found; render may fall back to CPU.")


def material(name: str, color: tuple[float, float, float, float], *, alpha: bool = False) -> bpy.types.Material:
    return cache_preview.make_material(name, color, roughness=0.32, alpha_blend=alpha, transmission=0.08 if alpha else 0.0)


def materials(config: dict) -> dict[str, bpy.types.Material]:
    container_cfg = config.get("container", {})
    return {
        "water": material("WaterParticles", tuple(config.get("particles", {}).get("color", [0.02, 0.78, 1.0, 1]))),
        "surface": material("WaterSurface", tuple(config.get("surface", {}).get("color", [0.18, 0.58, 1, 0.46])), alpha=True),
        "cloth": material("ClothSheet", (0.96, 0.80, 0.28, 1.0)),
        "solid_0": material("SolidLight", (1.00, 0.64, 0.18, 1.0)),
        "solid_1": material("SolidMid", (0.35, 0.82, 0.28, 1.0)),
        "solid_2": material("SolidDark", (0.04, 0.72, 0.88, 1.0)),
        "static": material("StaticBoundary", tuple(container_cfg.get("static_color", [0.78, 0.84, 0.90, 0.0])), alpha=True),
        "glass": material("CleanGlass", tuple(container_cfg.get("glass_color", [0.96, 0.99, 1.0, 0.0])), alpha=True),
        "floor": material("CleanFloor", tuple(container_cfg.get("floor_color", [0.82, 0.82, 0.80, 1.0]))),
        "wire": material("GlassEdges", tuple(container_cfg.get("edge_color", [0.0, 0.0, 0.0, 0.92])), alpha=True),
    }


def keyframe_visibility(obj: bpy.types.Object, *, visible_frame: int, frame_start: int, frame_end: int) -> None:
    surface_preview.keyframe_visibility(obj, visible_frame=visible_frame, frame_start=frame_start, frame_end=frame_end)


def set_node_value(node: bpy.types.Node, socket_names: tuple[str, ...], value) -> None:
    for socket_name in socket_names:
        socket = node.inputs.get(socket_name)
        if socket is not None:
            socket.default_value = value
            return


def get_particle_spheres_group(radius: float, mat: bpy.types.Material, config: dict) -> bpy.types.NodeTree:
    particles_cfg = config.get("particles", {})
    group = bpy.data.node_groups.new(f"FSIParticleSpheres_{radius:.6f}", "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    set_node_value(mesh_to_points, ("Radius",), radius)
    sphere = nodes.new("GeometryNodeMeshUVSphere")
    set_node_value(sphere, ("Segments",), int(particles_cfg.get("sphere_segments", 12)))
    set_node_value(sphere, ("Rings",), int(particles_cfg.get("sphere_rings", 6)))
    set_node_value(sphere, ("Radius",), radius)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    links.new(group_in.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], instance.inputs["Points"])
    links.new(sphere.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], set_mat.inputs["Geometry"])
    links.new(set_mat.outputs["Geometry"], group_out.inputs["Geometry"])
    return group


def add_particle_frame(frame: dict, metadata: dict, config: dict, mat: bpy.types.Material, frame_index: int, start: int, end: int) -> None:
    positions = np.asarray(frame.get("fluid_positions", []), dtype=np.float32).reshape(-1, 3)
    if positions.size == 0:
        return
    max_count = int(config.get("particles", {}).get("max_preview_particles", 25000))
    indices = cache_preview.sampled_indices(len(positions), max_count)
    positions = cache_preview.to_blender_positions(positions[indices], metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    mesh = bpy.data.meshes.new(f"WaterParticles_{frame_index:04d}")
    mesh.from_pydata([tuple(float(v) for v in point) for point in positions], [], [])
    obj = bpy.data.objects.new(f"WaterParticles_{frame_index:04d}", mesh)
    bpy.context.collection.objects.link(obj)
    radii = np.asarray(frame.get("fluid_radii", []), dtype=np.float32).reshape(-1)
    base_radius = float(np.median(radii[indices])) if radii.size >= len(indices) else float(metadata.get("fluid", {}).get("particle_radius", 0.006))
    modifier = obj.modifiers.new(name="ParticleSpheres", type="NODES")
    modifier.node_group = get_particle_spheres_group(base_radius * float(config.get("particles", {}).get("radius_scale", 0.75)), mat, config)
    keyframe_visibility(obj, visible_frame=frame_index, frame_start=start, frame_end=end)


def add_cloth_frame(frame: dict, metadata: dict, mat: bpy.types.Material, frame_index: int, start: int, end: int) -> None:
    cloth = metadata.get("cloth", {})
    indices = np.asarray(cloth.get("indices", []), dtype=np.int32).reshape(-1, 3)
    positions = np.asarray(frame.get("cloth_positions", []), dtype=np.float32).reshape(-1, 3)
    if positions.size == 0 or indices.size == 0:
        return
    positions = cache_preview.to_blender_positions(positions, metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    mesh = bpy.data.meshes.new(f"Cloth_{frame_index:04d}")
    mesh.from_pydata([tuple(float(v) for v in point) for point in positions], [], [tuple(int(v) for v in tri) for tri in indices])
    mesh.update()
    obj = bpy.data.objects.new(f"Cloth_{frame_index:04d}", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    keyframe_visibility(obj, visible_frame=frame_index, frame_start=start, frame_end=end)


def set_transform(obj: bpy.types.Object, transform: np.ndarray, axis_mode: str) -> None:
    obj.location = cache_preview.to_blender_position(transform[:3], axis_mode)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = cache_preview.to_blender_quaternion_wxyz(transform[3:7], axis_mode)


def create_shape_object(record: dict, mat: bpy.types.Material, axis_mode: str) -> bpy.types.Object | None:
    shape_type = str(record.get("type_name", ""))
    scale = tuple(float(v) for v in record.get("scale", [1, 1, 1]))
    if shape_type == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.dimensions = cache_preview.to_blender_scale((2.0 * scale[0], 2.0 * scale[1], 2.0 * scale[2]), axis_mode)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=max(scale[0], 1.0e-5))
        obj = bpy.context.object
    elif shape_type == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
        obj = bpy.context.object
        obj.scale = cache_preview.to_blender_scale(scale, axis_mode)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    else:
        return None
    obj.name = str(record.get("label", f"shape_{record.get('index', 0)}"))
    obj.data.materials.append(mat)
    return obj


def add_shapes(frames: list[dict], metadata: dict, mats: dict[str, bpy.types.Material], frame_indices: list[int]) -> None:
    shapes = metadata.get("shapes", [])
    if not shapes or not frames:
        return
    axis_mode = metadata.get("axis_conversion", "newton-y-up-to-blender-z-up")
    for local_index, record in enumerate(shapes):
        shape_index = int(record.get("index", local_index))
        body = int(record.get("body", -1))
        mat = mats["static"] if body < 0 else mats[f"solid_{body % 3}"]
        obj = create_shape_object(record, mat, axis_mode)
        if obj is None:
            continue
        for frame, frame_index in zip(frames, frame_indices, strict=True):
            transforms = np.asarray(frame.get("shape_world_q", []), dtype=np.float32).reshape(-1, 7)
            if shape_index >= len(transforms):
                continue
            set_transform(obj, transforms[shape_index], axis_mode)
            obj.keyframe_insert(data_path="location", frame=frame_index)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)


def add_wireframes(metadata: dict, mat: bpy.types.Material) -> None:
    axis_mode = metadata.get("axis_conversion", "newton-y-up-to-blender-z-up")
    for wire in metadata.get("wireframes", []):
        starts = np.asarray(wire.get("starts", []), dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(wire.get("ends", []), dtype=np.float32).reshape(-1, 3)
        if starts.size == 0 or starts.shape != ends.shape:
            continue
        starts = cache_preview.to_blender_positions(starts, axis_mode)
        ends = cache_preview.to_blender_positions(ends, axis_mode)
        curve = bpy.data.curves.new(str(wire.get("name", "wire")), type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.0015
        for start, end in zip(starts, ends, strict=True):
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            spline.points[0].co = (float(start[0]), float(start[1]), float(start[2]), 1.0)
            spline.points[1].co = (float(end[0]), float(end[1]), float(end[2]), 1.0)
        obj = bpy.data.objects.new(str(wire.get("name", "wire")), curve)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)


def container_bounds(metadata: dict) -> tuple[np.ndarray, np.ndarray] | None:
    axis_mode = metadata.get("axis_conversion", "newton-y-up-to-blender-z-up")
    tank = metadata.get("tank")
    if tank:
        half_width = float(tank.get("half_width", 1.0))
        half_depth = float(tank.get("half_depth", 1.0))
        floor_y = float(tank.get("floor_y", 0.0))
        top_y = float(tank.get("top_y", 1.0))
        points = np.asarray(
            [
                [-half_width, floor_y, -half_depth],
                [half_width, top_y, half_depth],
            ],
            dtype=np.float32,
        )
        converted = cache_preview.to_blender_positions(points, axis_mode)
        return converted.min(axis=0), converted.max(axis=0)

    points: list[np.ndarray] = []
    for wire in metadata.get("wireframes", []):
        starts = np.asarray(wire.get("starts", []), dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(wire.get("ends", []), dtype=np.float32).reshape(-1, 3)
        if starts.size:
            points.append(starts)
        if ends.size:
            points.append(ends)
    if not points:
        return None
    converted = cache_preview.to_blender_positions(np.concatenate(points, axis=0), axis_mode)
    return converted.min(axis=0), converted.max(axis=0)


def add_clean_floor_and_glass(metadata: dict, config: dict, mats: dict[str, bpy.types.Material]) -> None:
    container_cfg = config.get("container", {})
    bounds = container_bounds(metadata)
    if bounds is None:
        return
    lower, upper = bounds
    center = 0.5 * (lower + upper)
    size = np.maximum(upper - lower, 1.0e-5)
    wall_t = float(container_cfg.get("wall_thickness", 0.01))

    if bool(container_cfg.get("render_floor", True)):
        floor_size = float(container_cfg.get("floor_size", 337.5))
        floor_thickness = float(container_cfg.get("floor_thickness", 0.012))
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(float(center[0]), float(center[1]), float(lower[2] - 0.5 * floor_thickness)))
        floor = bpy.context.object
        floor.name = "CleanFloor"
        floor.dimensions = (floor_size, floor_size, floor_thickness)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        floor.data.materials.append(mats["floor"])

    if not bool(container_cfg.get("render_glass", True)):
        return

    glass = mats["glass"]
    panes = [
        ("GlassLeft", (lower[0] - 0.5 * wall_t, center[1], center[2]), (wall_t, size[1], size[2])),
        ("GlassRight", (upper[0] + 0.5 * wall_t, center[1], center[2]), (wall_t, size[1], size[2])),
        ("GlassBack", (center[0], lower[1] - 0.5 * wall_t, center[2]), (size[0], wall_t, size[2])),
        ("GlassFront", (center[0], upper[1] + 0.5 * wall_t, center[2]), (size[0], wall_t, size[2])),
    ]
    for name, location, dimensions in panes:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(float(v) for v in location))
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = tuple(float(v) for v in dimensions)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(glass)


def add_camera(metadata: dict) -> None:
    camera_cfg = metadata.get("camera", {}) or {}
    location = tuple(float(v) for v in camera_cfg.get("location", [1.5, -2.0, 1.2]))
    target = tuple(float(v) for v in camera_cfg.get("target", [0.0, 0.0, 0.55]))
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "FSIExperimentCamera"
    camera_type = str(camera_cfg.get("type", "PERSP")).upper()
    camera.data.type = "ORTHO" if camera_type == "ORTHO" else "PERSP"
    if camera.data.type == "ORTHO":
        camera.data.ortho_scale = float(camera_cfg.get("ortho_scale", 1.8))
    else:
        camera.data.lens = float(camera_cfg.get("focal_length_mm", camera_cfg.get("lens", 50.0)))
        camera.data.sensor_width = float(camera_cfg.get("sensor_width", 36.0))
    if "rotation_euler" in camera_cfg:
        camera.rotation_euler = tuple(float(v) for v in camera_cfg["rotation_euler"])
    else:
        cache_preview.look_at(camera, target)
    bpy.context.scene.camera = camera


def add_lighting(config: dict) -> None:
    lighting_cfg = config.get("lighting", {})
    for name, light_cfg in lighting_cfg.items():
        if name == "enabled" or not bool(light_cfg.get("enabled", True)):
            continue
        light_type = str(light_cfg.get("type", "AREA")).upper()
        bpy.ops.object.light_add(type=light_type, location=tuple(float(v) for v in light_cfg.get("location", [0.0, 0.0, 8.0])))
        light = bpy.context.object
        light.name = str(light_cfg.get("name", name))
        light.data.energy = float(light_cfg.get("energy", 500.0))
        if hasattr(light.data, "size") and "size" in light_cfg:
            light.data.size = float(light_cfg["size"])
        if "color" in light_cfg:
            light.data.color = tuple(float(v) for v in light_cfg["color"])
        if "rotation_euler" in light_cfg:
            light.rotation_euler = tuple(float(v) for v in light_cfg["rotation_euler"])
        elif "target" in light_cfg:
            cache_preview.look_at(light, tuple(float(v) for v in light_cfg["target"]))


def surface_records(surface_dir: Path, cache_dir: Path, frame_indices: list[int]) -> dict[int, Path]:
    particles_meta = load_json(cache_dir / "particles" / "splashsurf_particles_metadata.json")
    mapping: dict[int, Path] = {}
    for item in particles_meta.get("frames", []):
        source_frame = int(item.get("source_frame", item.get("export_index", 0)))
        export_index = int(item.get("export_index", source_frame))
        path = surface_dir / f"frame_{export_index:04d}.obj"
        if path.exists():
            mapping[source_frame] = path
    if not mapping:
        for frame_index in frame_indices:
            path = surface_dir / f"frame_{frame_index:04d}.obj"
            if path.exists():
                mapping[frame_index] = path
    return mapping


def add_surface_frames(surface_dir: Path, cache_dir: Path, frame_indices: list[int], mat: bpy.types.Material, start: int, end: int) -> None:
    if not surface_dir.exists():
        raise FileNotFoundError(f"Surface OBJ directory does not exist: {surface_dir}")
    collection = surface_preview.create_surface_collection("Water_Surface_Sequence")
    mapping = surface_records(surface_dir, cache_dir, frame_indices)
    if not mapping:
        raise FileNotFoundError(f"No splashsurf OBJ frames found in {surface_dir}")
    for frame_index in frame_indices:
        path = mapping.get(frame_index)
        if path is None:
            continue
        imported = surface_preview.import_surface_frame(path, collection, mat)
        for index, obj in enumerate(imported):
            obj.name = f"WaterSurface_{frame_index:04d}_{index:02d}"
            keyframe_visibility(obj, visible_frame=frame_index, frame_start=start, frame_end=end)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    metadata = load_json(args.cache_dir / "metadata.json")
    frame_indices = parse_frame_indices(args.frame_indices, metadata, args.cache_dir)
    frames = [read_frame(args.cache_dir, frame_index) for frame_index in frame_indices]
    frame_start = min(frame_indices) if frame_indices else 1
    frame_end = max(frame_indices) if frame_indices else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_preview.clear_scene()
    configure_scene(config, metadata, args.output_dir)
    mats = materials(config)
    add_clean_floor_and_glass(metadata, config, mats)
    add_wireframes(metadata, mats["wire"])
    add_shapes(frames, metadata, mats, frame_indices)
    for frame, frame_index in zip(frames, frame_indices, strict=True):
        add_cloth_frame(frame, metadata, mats["cloth"], frame_index, frame_start, frame_end)
        if args.mode == "particles":
            add_particle_frame(frame, metadata, config, mats["water"], frame_index, frame_start, frame_end)
    if args.mode == "surface":
        surface_dir = args.surface_dir or args.cache_dir / "surface_obj"
        add_surface_frames(surface_dir, args.cache_dir, frame_indices, mats["surface"], frame_start, frame_end)
    add_camera(metadata)
    add_lighting(config)

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)
    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))
    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
