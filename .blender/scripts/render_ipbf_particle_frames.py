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


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render selected IPBF cached particle frames in Blender.")
    parser.add_argument("--config", type=Path, required=True, help="IPBF particle-example config JSON.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Optional override for cache root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for rendered frame directory.")
    parser.add_argument("--case-key", type=str, default=None, help="Render only one case key.")
    parser.add_argument("--frame-indices", type=str, default=None, help="Comma-separated frame indices overriding config.")
    parser.add_argument(
        "--save-blend",
        type=Path,
        default=None,
        help="Optional .blend path for one case, or output directory for per-case .blend templates.",
    )
    parser.add_argument("--render", action="store_true", help="Render PNG stills. Without this flag, only builds the first frame.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(config_value: str | None, override: Path | None) -> Path | None:
    if override is not None:
        return override.resolve()
    if config_value is None:
        return None
    return Path(config_value).resolve()


def configure_scene(config: dict, output_path: Path) -> None:
    render_cfg = config.get("render", {})
    resolution = render_cfg.get("resolution", [1920, 1080])
    scene = bpy.context.scene
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.fps = int(render_cfg.get("fps", 60))

    engine = str(render_cfg.get("engine", "BLENDER_EEVEE_NEXT"))
    try:
        scene.render.engine = engine
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if scene.render.engine == "CYCLES":
        configure_cycles_device(render_cfg)
        scene.cycles.samples = int(render_cfg.get("samples", 96))
        scene.cycles.use_denoising = bool(render_cfg.get("denoise", True))
        scene.cycles.max_bounces = int(render_cfg.get("max_bounces", 8))
        scene.cycles.transparent_max_bounces = int(render_cfg.get("transparent_max_bounces", 8))
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(render_cfg.get("samples", 16))
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = bool(render_cfg.get("use_raytracing", True))
        if hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = bool(render_cfg.get("use_ssr", True))
        if hasattr(scene.eevee, "use_ssr_refraction"):
            scene.eevee.use_ssr_refraction = bool(render_cfg.get("use_ssr_refraction", True))
    if hasattr(scene, "eevee_next"):
        scene.eevee_next.taa_render_samples = int(render_cfg.get("samples", 16))

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        color = render_cfg.get("background", [0.0, 0.0, 0.0, 1.0])
        cache_preview.set_node_input(bg, "Color", tuple(float(v) for v in color))
        cache_preview.set_node_input(bg, "Strength", float(render_cfg.get("background_strength", 1.0)))

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = float(render_cfg.get("exposure", 0.0))
        scene.view_settings.gamma = float(render_cfg.get("gamma", 1.0))
    except Exception:
        pass


def configure_cycles_device(render_cfg: dict) -> None:
    scene = bpy.context.scene
    scene.cycles.device = str(render_cfg.get("device", "GPU")).upper()
    if scene.cycles.device != "GPU":
        return

    preferred = [str(render_cfg.get("compute_device_type", "OPTIX")).upper(), "CUDA", "HIP", "ONEAPI", "METAL"]
    preferences = bpy.context.preferences.addons.get("cycles")
    if preferences is None:
        print("[IPBF Particles] Cycles preferences not found; render may fall back to CPU.")
        return

    cycles_preferences = preferences.preferences
    selected_type = None
    for device_type in dict.fromkeys(preferred):
        try:
            cycles_preferences.compute_device_type = device_type
            cycles_preferences.get_devices()
        except Exception:
            continue
        devices = list(getattr(cycles_preferences, "devices", []))
        gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
        if gpu_devices:
            selected_type = device_type
            for device in devices:
                device.use = device in gpu_devices
            break

    if selected_type is None:
        print("[IPBF Particles] No Cycles GPU device found; render may fall back to CPU.")
    else:
        print(f"[IPBF Particles] Cycles GPU device type: {selected_type}")


def disable_shadows(obj: bpy.types.Object) -> None:
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False


def make_principled_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.35,
    alpha_blend: bool = False,
    transmission: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    material.blend_method = "BLEND" if alpha_blend else "OPAQUE"
    material.use_screen_refraction = alpha_blend
    material.show_transparent_back = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        cache_preview.set_node_input(bsdf, "Base Color", color)
        cache_preview.set_node_input(bsdf, "Alpha", float(color[3]))
        cache_preview.set_node_input(bsdf, "Roughness", roughness)
        cache_preview.set_node_input(bsdf, "Metallic", 0.0)
        cache_preview.set_node_input(bsdf, "Transmission Weight", transmission)
        cache_preview.set_node_input(bsdf, "Transmission", transmission)
    return material


def make_emission_material(name: str, color: tuple[float, float, float, float], strength: float = 1.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_particle_material(name: str, config: dict):
    particles_cfg = config.get("particles", {})
    color_mode = str(particles_cfg.get("color_mode", "uniform"))
    base_color = tuple(float(v) for v in particles_cfg.get("uniform_color", [0.02, 0.36, 0.95]))
    material_color = (base_color[0], base_color[1], base_color[2], 1.0)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = material_color
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        cache_preview.set_node_input(bsdf, "Base Color", material_color)
        cache_preview.set_node_input(bsdf, "Roughness", float(particles_cfg.get("roughness", 0.32)))
        cache_preview.set_node_input(bsdf, "Alpha", 1.0)
        cache_preview.set_node_input(bsdf, "Specular IOR Level", float(particles_cfg.get("specular", 0.65)))
        cache_preview.set_node_input(bsdf, "Coat Weight", float(particles_cfg.get("coat_weight", 0.0)))
        if color_mode == "density":
            attribute = nodes.new("ShaderNodeAttribute")
            attribute.attribute_name = "density_color"
            links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def density_ratio_to_color(ratio: np.ndarray, min_ratio: float, max_ratio: float) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=np.float32)
    t = np.clip((ratio - min_ratio) / max(max_ratio - min_ratio, 1.0e-8), 0.0, 1.0)[:, None]
    low = np.array([0.00, 0.55, 1.00], dtype=np.float32)
    mid = np.array([0.05, 0.95, 0.90], dtype=np.float32)
    high = np.array([1.00, 0.18, 0.05], dtype=np.float32)
    colors = np.where(t < 0.5, low + (mid - low) * (2.0 * t), mid + (high - mid) * (2.0 * (t - 0.5)))
    return colors.astype(np.float32, copy=False)


def set_node_value(node: bpy.types.Node, socket_names: tuple[str, ...], value) -> None:
    for socket_name in socket_names:
        socket = node.inputs.get(socket_name)
        if socket is not None:
            socket.default_value = value
            return


def get_particle_spheres_group(radius: float, material: bpy.types.Material, config: dict) -> bpy.types.NodeTree:
    particles_cfg = config.get("particles", {})
    name = f"IPBFParticleSpheres_{radius:.6f}"
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    set_node_value(mesh_to_points, ("Radius",), float(radius))
    sphere = nodes.new("GeometryNodeMeshUVSphere")
    set_node_value(sphere, ("Segments",), int(particles_cfg.get("sphere_segments", 12)))
    set_node_value(sphere, ("Rings",), int(particles_cfg.get("sphere_rings", 6)))
    set_node_value(sphere, ("Radius",), float(radius))
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], instance.inputs["Points"])
    links.new(sphere.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])
    return group


def create_particle_object(
    *,
    name: str,
    payload: dict[str, np.ndarray],
    metadata: dict,
    config: dict,
    material: bpy.types.Material,
) -> bpy.types.Object:
    axis_mode = str(metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    positions = cache_preview.to_blender_positions(np.asarray(payload["fluid_positions"], dtype=np.float32), axis_mode)
    particles_cfg = config.get("particles", {})
    color_mode = str(particles_cfg.get("color_mode", "density"))
    if color_mode == "uniform" or "fluid_density" not in payload:
        base = np.asarray(particles_cfg.get("uniform_color", [0.05, 0.55, 1.0]), dtype=np.float32)
        colors = np.repeat(base[None, :], positions.shape[0], axis=0)
    else:
        density = np.asarray(payload["fluid_density"], dtype=np.float32).reshape(-1)
        ratio = density / max(float(metadata.get("rest_density", 1000.0)), 1.0e-8)
        colors = density_ratio_to_color(
            ratio,
            float(particles_cfg.get("density_color_min_ratio", 0.90)),
            float(particles_cfg.get("density_color_max_ratio", 1.30)),
        )

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(float(v) for v in point) for point in positions], [], [])
    color_attr = mesh.color_attributes.new(name="density_color", type="FLOAT_COLOR", domain="POINT")
    rgba = np.empty((len(colors), 4), dtype=np.float32)
    rgba[:, :3] = colors
    rgba[:, 3] = 1.0
    color_attr.data.foreach_set("color", rgba.reshape(-1))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    radius = float(metadata.get("particle_display_radius", 0.005)) * float(particles_cfg.get("radius_scale", 0.80))
    modifier = obj.modifiers.new(name="ParticleSpheres", type="NODES")
    modifier.node_group = get_particle_spheres_group(radius, material, config)
    return obj


def add_wireframe(metadata: dict, material: bpy.types.Material) -> None:
    starts = np.asarray(metadata.get("wireframe_starts", []), dtype=np.float32)
    ends = np.asarray(metadata.get("wireframe_ends", []), dtype=np.float32)
    if starts.size == 0 or ends.size == 0 or starts.shape != ends.shape:
        return
    axis_mode = str(metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    starts_b = cache_preview.to_blender_positions(starts, axis_mode)
    ends_b = cache_preview.to_blender_positions(ends, axis_mode)
    curve_data = bpy.data.curves.new("IPBFContainerWireframe", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.0015
    curve_data.fill_mode = "FULL"
    for start, end in zip(starts_b, ends_b, strict=True):
        spline = curve_data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (float(start[0]), float(start[1]), float(start[2]), 1.0)
        spline.points[1].co = (float(end[0]), float(end[1]), float(end[2]), 1.0)
    obj = bpy.data.objects.new("IPBFContainerWireframe", curve_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    disable_shadows(obj)


def add_glass_container(metadata: dict, config: dict, materials: dict[str, bpy.types.Material]) -> None:
    container_cfg = config.get("container", {})
    tank = metadata.get("tank", {})
    axis_mode = str(metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    half_width = float(tank.get("half_width", 1.0))
    half_depth = float(tank.get("half_depth", 1.0))
    floor_y = float(tank.get("floor_y", 0.0))
    top_y = float(tank.get("top_y", 1.0))
    height = max(top_y - floor_y, 1.0e-5)
    center_y = floor_y + 0.5 * height
    wall_t = float(container_cfg.get("wall_thickness", 0.01))

    if bool(container_cfg.get("render_floor", True)):
        extent_scale = float(container_cfg.get("floor_extent_scale", 2.2))
        floor_thickness = float(container_cfg.get("floor_thickness", 0.012))
        cache_preview.add_box(
            "CleanFloor",
            (0.0, floor_y - 0.5 * floor_thickness, 0.0),
            (2.0 * half_width * extent_scale, floor_thickness, 2.0 * half_depth * extent_scale),
            materials["floor"],
            axis_mode,
        )

    if not bool(container_cfg.get("render_glass", True)):
        return

    glass = materials["glass"]
    panes = [
        ("GlassLeft", (-(half_width + 0.5 * wall_t), center_y, 0.0), (wall_t, height, 2.0 * half_depth)),
        ("GlassRight", ((half_width + 0.5 * wall_t), center_y, 0.0), (wall_t, height, 2.0 * half_depth)),
        ("GlassBack", (0.0, center_y, -(half_depth + 0.5 * wall_t)), (2.0 * half_width, height, wall_t)),
        ("GlassFront", (0.0, center_y, (half_depth + 0.5 * wall_t)), (2.0 * half_width, height, wall_t)),
    ]
    for name, location, scale in panes:
        cache_preview.add_box(name, location, scale, glass, axis_mode)
        obj = bpy.context.object
        disable_shadows(obj)


def camera_config(case_config: dict, metadata: dict) -> dict:
    if "camera" in case_config:
        return dict(case_config["camera"])
    tank = metadata.get("tank", {})
    top_y = float(tank.get("top_y", 1.0))
    half_width = float(tank.get("half_width", 1.0))
    return {
        "location": [1.8 * half_width, -2.5 * half_width, 0.65 * top_y],
        "target": [0.0, 0.0, 0.5 * top_y],
        "ortho_scale": max(2.2 * half_width, 1.25 * top_y),
    }


def add_camera(case_config: dict, metadata: dict) -> None:
    camera_cfg = camera_config(case_config, metadata)
    bpy.ops.object.camera_add(location=tuple(float(v) for v in camera_cfg["location"]))
    camera = bpy.context.object
    camera.name = "IPBFParticleCamera"
    camera_type = str(camera_cfg.get("type", "PERSP")).upper()
    camera.data.type = "ORTHO" if camera_type == "ORTHO" else "PERSP"
    if camera.data.type == "ORTHO":
        camera.data.ortho_scale = float(camera_cfg.get("ortho_scale", 2.0))
    else:
        camera.data.lens = float(camera_cfg.get("focal_length_mm", camera_cfg.get("lens", 50.0)))
        camera.data.sensor_width = float(camera_cfg.get("sensor_width", 36.0))
    if "rotation_euler" in camera_cfg:
        camera.rotation_euler = tuple(float(v) for v in camera_cfg["rotation_euler"])
    else:
        cache_preview.look_at(camera, tuple(float(v) for v in camera_cfg["target"]))
    camera.data.clip_end = float(camera_cfg.get("clip_end", 1000.0))
    bpy.context.scene.camera = camera


def add_lighting(config: dict) -> None:
    lighting_cfg = config.get("lighting", {})
    key_cfg = lighting_cfg.get("key", {})
    location = tuple(float(v) for v in key_cfg.get("location", [2.4, -3.3, 3.4]))
    target = tuple(float(v) for v in key_cfg.get("target", [-0.45, 0.45, 0.25]))
    bpy.ops.object.light_add(type="AREA", location=location)
    key = bpy.context.object
    key.name = "KeyLight_RightFrontTop"
    key.data.energy = float(key_cfg.get("energy", 850.0))
    key.data.size = float(key_cfg.get("size", 2.1))
    cache_preview.look_at(key, target)

    fill_cfg = lighting_cfg.get("fill", {})
    if bool(fill_cfg.get("enabled", True)):
        bpy.ops.object.light_add(type="AREA", location=tuple(float(v) for v in fill_cfg.get("location", [-2.0, 1.8, 2.0])))
        fill = bpy.context.object
        fill.name = "SoftFill"
        fill.data.energy = float(fill_cfg.get("energy", 55.0))
        fill.data.size = float(fill_cfg.get("size", 5.0))
        cache_preview.look_at(fill, tuple(float(v) for v in fill_cfg.get("target", [0.0, 0.0, 0.55])))


def available_frame_indices(case_dir: Path) -> list[int]:
    frame_dir = case_dir / "frames"
    paths = sorted(frame_dir.glob("frame_*.npz"))
    indices: list[int] = []
    for path in paths:
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return indices


def parse_frame_indices(raw: str | None, case_config: dict, case_dir: Path) -> list[int]:
    if raw and raw.strip().lower() == "auto":
        indices = available_frame_indices(case_dir)
        if not indices:
            raise FileNotFoundError(f"No cached frames found in {case_dir / 'frames'}.")
        return indices
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(value) for value in case_config.get("render_frames", [0])]


def frame_payload(case_dir: Path, frame_index: int) -> dict[str, np.ndarray]:
    path = case_dir / "frames" / f"frame_{frame_index:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame cache not found: {path}")
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def set_visible_only_at_frame(obj: bpy.types.Object, *, visible_frame: int, frame_start: int, frame_end: int) -> None:
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_start)
    obj.keyframe_insert(data_path="hide_render", frame=frame_start)

    if visible_frame > frame_start:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=visible_frame - 1)
        obj.keyframe_insert(data_path="hide_render", frame=visible_frame - 1)

    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=visible_frame)
    obj.keyframe_insert(data_path="hide_render", frame=visible_frame)

    if visible_frame < frame_end:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=visible_frame + 1)
        obj.keyframe_insert(data_path="hide_render", frame=visible_frame + 1)


def blend_path_for_case(save_blend: Path, case_config: dict, multiple_cases: bool) -> Path:
    key = str(case_config["key"])
    resolved = save_blend.resolve()
    if resolved.suffix.lower() == ".blend":
        if multiple_cases:
            return resolved.with_name(f"{resolved.stem}_{key}.blend")
        return resolved
    return resolved / f"{key}.blend"


def build_case_template(
    *,
    config: dict,
    case_config: dict,
    case_dir: Path,
    output_dir: Path,
    frame_indices: list[int],
    save_blend: Path,
) -> None:
    if not frame_indices:
        raise ValueError(f"No frames selected for {case_config['key']}.")

    metadata = load_json(case_dir / "metadata.json")
    frame_start = min(frame_indices)
    frame_end = max(frame_indices)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_preview.clear_scene()
    configure_scene(config, output_dir / f"{case_config['key']}_frame_")
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)

    materials = {
        "particles": make_particle_material("IPBFParticles", config),
        "glass": make_principled_material(
            "CleanGlass",
            tuple(float(v) for v in config.get("container", {}).get("glass_color", [0.94, 0.98, 1.0, 0.12])),
            roughness=float(config.get("container", {}).get("glass_roughness", 0.02)),
            alpha_blend=True,
            transmission=float(config.get("container", {}).get("glass_transmission", 0.35)),
        ),
        "floor": make_principled_material(
            "CleanFloor",
            tuple(float(v) for v in config.get("container", {}).get("floor_color", [0.86, 0.86, 0.84, 1.0])),
            roughness=float(config.get("container", {}).get("floor_roughness", 0.52)),
        ),
        "wire": make_principled_material(
            "GlassEdges",
            tuple(float(v) for v in config.get("container", {}).get("edge_color", [0.02, 0.025, 0.03, 0.85])),
            roughness=0.25,
            alpha_blend=True,
        ),
    }

    add_glass_container(metadata, config, materials)

    for frame_index in frame_indices:
        payload = frame_payload(case_dir, frame_index)
        obj = create_particle_object(
            name=f"{case_config['key']}_particles_{frame_index:04d}",
            payload=payload,
            metadata=metadata,
            config=config,
            material=materials["particles"],
        )
        set_visible_only_at_frame(obj, visible_frame=frame_index, frame_start=frame_start, frame_end=frame_end)

    if bool(config.get("render_wireframe", True)):
        add_wireframe(metadata, materials["wire"])
    add_camera(case_config, metadata)
    add_lighting(config)

    save_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))


def render_case_frame(
    *,
    config: dict,
    case_config: dict,
    case_dir: Path,
    output_dir: Path,
    frame_index: int,
    render: bool,
) -> None:
    metadata = load_json(case_dir / "metadata.json")
    payload = frame_payload(case_dir, frame_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_config['key']}_frame_{frame_index:04d}.png"

    cache_preview.clear_scene()
    configure_scene(config, output_path)
    materials = {
        "particles": make_particle_material("IPBFParticles", config),
        "glass": make_principled_material(
            "CleanGlass",
            tuple(float(v) for v in config.get("container", {}).get("glass_color", [0.94, 0.98, 1.0, 0.12])),
            roughness=float(config.get("container", {}).get("glass_roughness", 0.02)),
            alpha_blend=True,
            transmission=float(config.get("container", {}).get("glass_transmission", 0.35)),
        ),
        "floor": make_principled_material(
            "CleanFloor",
            tuple(float(v) for v in config.get("container", {}).get("floor_color", [0.86, 0.86, 0.84, 1.0])),
            roughness=float(config.get("container", {}).get("floor_roughness", 0.52)),
        ),
        "wire": make_principled_material(
            "GlassEdges",
            tuple(float(v) for v in config.get("container", {}).get("edge_color", [0.02, 0.025, 0.03, 0.85])),
            roughness=0.25,
            alpha_blend=True,
        ),
    }
    add_glass_container(metadata, config, materials)
    create_particle_object(
        name=f"{case_config['key']}_particles_{frame_index:04d}",
        payload=payload,
        metadata=metadata,
        config=config,
        material=materials["particles"],
    )
    if bool(config.get("render_wireframe", True)):
        add_wireframe(metadata, materials["wire"])
    add_camera(case_config, metadata)
    add_lighting(config)

    if render:
        bpy.context.scene.frame_set(1)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    cache_root = resolve_path(config.get("cache_dir"), args.cache_root)
    output_dir = resolve_path(config.get("render_dir"), args.output_dir)
    if cache_root is None or output_dir is None:
        raise ValueError("Config must define cache_dir and render_dir unless overrides are supplied.")

    cases = [case for case in config["cases"] if args.case_key is None or str(case["key"]) == args.case_key]
    if not cases:
        raise KeyError(f"Case key not found: {args.case_key}")

    for case_config in cases:
        case_dir = cache_root / str(case_config["key"])
        frame_indices = parse_frame_indices(args.frame_indices, case_config, case_dir)
        if args.save_blend is not None:
            blend_path = blend_path_for_case(args.save_blend, case_config, multiple_cases=len(cases) > 1)
            print(f"[IPBF Particles] Save template {blend_path}")
            build_case_template(
                config=config,
                case_config=case_config,
                case_dir=case_dir,
                output_dir=output_dir,
                frame_indices=frame_indices,
                save_blend=blend_path,
            )
        if args.render or args.save_blend is None:
            for frame_index in frame_indices:
                print(f"[IPBF Particles] Render {case_config['key']} frame {frame_index}")
                render_case_frame(
                    config=config,
                    case_config=case_config,
                    case_dir=case_dir,
                    output_dir=output_dir,
                    frame_index=frame_index,
                    render=args.render,
                )


if __name__ == "__main__":
    main()
