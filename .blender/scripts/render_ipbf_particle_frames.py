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
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(render_cfg.get("samples", 16))
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


def disable_shadows(obj: bpy.types.Object) -> None:
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False


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


def make_particle_material(name: str):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "density_color"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Strength"].default_value = 1.4
    links.new(attribute.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def density_ratio_to_color(ratio: np.ndarray, min_ratio: float, max_ratio: float) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=np.float32)
    t = np.clip((ratio - min_ratio) / max(max_ratio - min_ratio, 1.0e-8), 0.0, 1.0)[:, None]
    low = np.array([0.00, 0.55, 1.00], dtype=np.float32)
    mid = np.array([0.05, 0.95, 0.90], dtype=np.float32)
    high = np.array([1.00, 0.18, 0.05], dtype=np.float32)
    colors = np.where(t < 0.5, low + (mid - low) * (2.0 * t), mid + (high - mid) * (2.0 * (t - 0.5)))
    return colors.astype(np.float32, copy=False)


def get_particle_points_group(radius: float, material: bpy.types.Material) -> bpy.types.NodeTree:
    name = f"IPBFParticlePoints_{radius:.6f}"
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    mesh_to_points.inputs["Radius"].default_value = float(radius)
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
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
    modifier = obj.modifiers.new(name="ParticlePoints", type="NODES")
    modifier.node_group = get_particle_points_group(radius, material)
    disable_shadows(obj)
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
    curve_data.bevel_depth = 0.0025
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
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = float(camera_cfg.get("ortho_scale", 2.0))
    cache_preview.look_at(camera, tuple(float(v) for v in camera_cfg["target"]))
    bpy.context.scene.camera = camera


def parse_frame_indices(raw: str | None, case_config: dict) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(value) for value in case_config.get("render_frames", [0])]


def frame_payload(case_dir: Path, frame_index: int) -> dict[str, np.ndarray]:
    path = case_dir / "frames" / f"frame_{frame_index:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame cache not found: {path}")
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


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
        "particles": make_particle_material("IPBFParticles"),
        "wire": make_emission_material("IPBFWire", (0.85, 0.90, 1.0, 1.0), 1.3),
    }
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
        for frame_index in parse_frame_indices(args.frame_indices, case_config):
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
