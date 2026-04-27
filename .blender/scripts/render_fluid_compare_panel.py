from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import render_fsi_cache as cache_preview


COLORBAR_MIN_RATIO = 0.90
COLORBAR_MAX_RATIO = 1.30
COLORBAR_RHO0_RATIO = 1.00
COLORBAR_RHO12_RATIO = 1.20


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Panel-render config JSON.")
    parser.add_argument("--panel-cache-dir", type=Path, default=None, help="Optional override for panel cache root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for render output.")
    parser.add_argument("--save-blend", type=Path, default=None, help="Optional override for .blend output.")
    parser.add_argument("--skip-save-blend", action="store_true", help="Do not save a .blend file even if config defines one.")
    parser.add_argument("--frames", type=int, default=None, help="Optional limit on rendered frame count.")
    parser.add_argument("--render", action="store_true", help="Render the animation after scene setup.")
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


def configure_scene(config: dict, frame_count: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    render_cfg = config.get("render", {})
    resolution = render_cfg.get("resolution", [2560, 900])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.filepath = str(output_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start = 1
    scene.frame_end = int(frame_count)
    scene.render.film_transparent = False
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(render_cfg.get("samples", 32))
    scene.cycles.use_denoising = True
    scene.cycles.device = "GPU"
    scene.render.fps = int(render_cfg.get("fps", 60))

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is not None:
        cycles_prefs = cycles_addon.preferences
        for compute_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                cycles_prefs.compute_device_type = compute_type
                cycles_prefs.get_devices()
                if any(getattr(device, "type", "") != "CPU" for device in cycles_prefs.devices):
                    for device in cycles_prefs.devices:
                        device.use = getattr(device, "type", "") != "CPU"
                    break
            except Exception:
                continue

    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = float(render_cfg.get("exposure", 0.20))
    scene.view_settings.gamma = float(render_cfg.get("gamma", 1.0))

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        cache_preview.set_node_input(bg, "Color", (0.0, 0.0, 0.0, 1.0))
        cache_preview.set_node_input(bg, "Strength", 1.0)


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
    emission.inputs["Strength"].default_value = 1.35
    links.new(attribute.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def density_ratio_to_color(ratio: float) -> tuple[float, float, float]:
    r = min(max(ratio, COLORBAR_MIN_RATIO), COLORBAR_MAX_RATIO)
    c_blue = np.array([0.00, 0.12, 1.00], dtype=np.float32)
    c_cyan = np.array([0.00, 0.88, 1.00], dtype=np.float32)
    c_green = np.array([0.00, 0.98, 0.10], dtype=np.float32)
    c_yellow = np.array([1.00, 0.96, 0.02], dtype=np.float32)
    c_red = np.array([1.00, 0.08, 0.04], dtype=np.float32)

    if r <= 1.00:
        color = c_blue
    elif r <= 1.05:
        t = (r - 1.00) / 0.05
        color = (1.0 - t) * c_blue + t * c_cyan
    elif r <= 1.10:
        t = (r - 1.05) / 0.05
        color = (1.0 - t) * c_cyan + t * c_green
    elif r <= 1.15:
        t = (r - 1.10) / 0.05
        color = (1.0 - t) * c_green + t * c_yellow
    elif r <= 1.20:
        t = (r - 1.15) / 0.05
        color = (1.0 - t) * c_yellow + t * c_red
    else:
        color = c_red
    return (float(color[0]), float(color[1]), float(color[2]))


def create_colorbar_image(
    name: str,
    width: int = 1024,
    height: int = 32,
    *,
    min_ratio: float = COLORBAR_MIN_RATIO,
    max_ratio: float = COLORBAR_MAX_RATIO,
) -> bpy.types.Image:
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "sRGB"
    pixels = np.zeros((height, width, 4), dtype=np.float32)
    for x in range(width):
        ratio = min_ratio + (max_ratio - min_ratio) * (float(x) / float(max(1, width - 1)))
        color = density_ratio_to_color(ratio)
        pixels[:, x, 0] = color[0]
        pixels[:, x, 1] = color[1]
        pixels[:, x, 2] = color[2]
        pixels[:, x, 3] = 1.0
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()
    image.pack()
    image.use_fake_user = True
    return image


def make_colorbar_material(name: str, image: bpy.types.Image):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Strength"].default_value = 1.35
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def get_particle_points_group(radius: float, material: bpy.types.Material) -> bpy.types.NodeTree:
    name = f"FluidComparePoints_{radius:.6f}"
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        return existing

    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = group.nodes
    links = group.links

    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-300, 0)
    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (240, 0)

    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-40, 0)
    mesh_to_points.mode = "VERTICES"
    mesh_to_points.inputs["Radius"].default_value = float(radius)

    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.location = (120, 0)
    set_material.inputs["Material"].default_value = material

    links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])
    return group


def keyframe_visibility(obj: bpy.types.Object, *, visible_frame: int, frame_start: int, frame_end: int) -> None:
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_start)
    obj.keyframe_insert(data_path="hide_render", frame=frame_start)
    if visible_frame > frame_start:
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


def add_case_outline(
    *,
    name: str,
    center_x: float,
    half_width: float,
    bottom_z: float,
    top_z: float,
    front_y: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.0025
    curve_data.fill_mode = "FULL"
    spline = curve_data.splines.new("POLY")
    spline.points.add(4)
    points = [
        (center_x - half_width, front_y, bottom_z, 1.0),
        (center_x + half_width, front_y, bottom_z, 1.0),
        (center_x + half_width, front_y, top_z, 1.0),
        (center_x - half_width, front_y, top_z, 1.0),
        (center_x - half_width, front_y, bottom_z, 1.0),
    ]
    for point, value in zip(spline.points, points, strict=True):
        point.co = value
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    disable_shadows(obj)
    return obj


def add_text_label(
    *,
    name: str,
    text: str,
    location: tuple[float, float, float],
    size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.object.text_add(location=location, rotation=(np.pi / 2.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.size = float(size)
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.materials.append(material)
    disable_shadows(obj)
    return obj


def add_colorbar(
    *,
    total_width: float,
    center_z: float,
    material: bpy.types.Material,
    text_material: bpy.types.Material,
    tick_material: bpy.types.Material,
    min_ratio: float = COLORBAR_MIN_RATIO,
    max_ratio: float = COLORBAR_MAX_RATIO,
) -> None:
    def ratio_to_x(ratio: float) -> float:
        t = (ratio - min_ratio) / max(max_ratio - min_ratio, 1.0e-8)
        return (t - 0.5) * total_width

    bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, center_z), rotation=(np.pi / 2.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = "Density_Colorbar"
    obj.scale = (0.5 * total_width, 0.03, 0.04)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    disable_shadows(obj)

    for name, ratio in (("Colorbar_Rho0_Tick", COLORBAR_RHO0_RATIO), ("Colorbar_Rho12_Tick", COLORBAR_RHO12_RATIO)):
        bpy.ops.mesh.primitive_plane_add(
            location=(ratio_to_x(ratio), -0.001, center_z),
            rotation=(np.pi / 2.0, 0.0, 0.0),
        )
        tick = bpy.context.object
        tick.name = name
        tick.scale = (0.003, 0.03, 0.060)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        tick.data.materials.append(tick_material)
        disable_shadows(tick)

    add_text_label(
        name="Colorbar_Rho0",
        text="\u03c10",
        location=(ratio_to_x(COLORBAR_RHO0_RATIO), 0.0, center_z - 0.075),
        size=0.070,
        material=text_material,
    )
    add_text_label(
        name="Colorbar_Rho12",
        text="1.2\u03c10",
        location=(ratio_to_x(COLORBAR_RHO12_RATIO), 0.0, center_z - 0.075),
        size=0.070,
        material=text_material,
    )


def create_particle_frame_object(
    *,
    name: str,
    positions: np.ndarray,
    colors: np.ndarray,
    x_offset: float,
    axis_mode: str,
    points_group: bpy.types.NodeTree,
) -> bpy.types.Object:
    positions = cache_preview.to_blender_positions(np.asarray(positions, dtype=np.float32), axis_mode)
    positions[:, 0] += float(x_offset)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(float(v) for v in point) for point in positions], [], [])
    color_attr = mesh.color_attributes.new(name="density_color", type="FLOAT_COLOR", domain="POINT")

    rgba = np.empty((len(colors), 4), dtype=np.float32)
    rgba[:, :3] = np.asarray(colors, dtype=np.float32)
    rgba[:, 3] = 1.0
    color_attr.data.foreach_set("color", rgba.reshape(-1))

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new(name="ParticlePoints", type="NODES")
    modifier.node_group = points_group
    disable_shadows(obj)
    return obj


def create_box_object(
    *,
    name: str,
    half_extent: list[float],
    material: bpy.types.Material,
    axis_mode: str,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = cache_preview.to_blender_scale(
        (2.0 * float(half_extent[0]), 2.0 * float(half_extent[1]), 2.0 * float(half_extent[2])),
        axis_mode,
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    disable_shadows(obj)
    return obj


def keyframe_box_animation(
    *,
    obj: bpy.types.Object,
    frame_payloads: list[dict[str, np.ndarray]],
    x_offset: float,
    axis_mode: str,
) -> None:
    obj.rotation_mode = "QUATERNION"
    for timeline_frame, payload in enumerate(frame_payloads, start=1):
        body_q = np.asarray(payload["box_body_q"], dtype=np.float32)
        if body_q.shape[0] == 0:
            continue
        position = body_q[0, :3].copy()
        position[0] += float(x_offset)
        obj.location = cache_preview.to_blender_position(position, axis_mode)
        obj.rotation_quaternion = cache_preview.to_blender_quaternion_wxyz(body_q[0, 3:7], axis_mode)
        obj.keyframe_insert(data_path="location", frame=timeline_frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=timeline_frame)


def load_case_frames(case_dir: Path, max_frames: int | None) -> list[dict[str, np.ndarray]]:
    frame_paths = sorted((case_dir / "frames").glob("frame_*.npz"))
    if max_frames is not None:
        frame_paths = frame_paths[: int(max_frames)]
    frames: list[dict[str, np.ndarray]] = []
    for path in frame_paths:
        with np.load(path, allow_pickle=True) as data:
            frames.append({key: data[key] for key in data.files})
    return frames


def add_case_panel(
    *,
    case_info: dict,
    center_x: float,
    frame_count: int,
    materials: dict[str, bpy.types.Material],
    layout: dict,
    axis_mode: str,
) -> None:
    case_dir = Path(case_info["path"])
    case_meta = load_json(case_dir / "metadata.json")
    frame_payloads = load_case_frames(case_dir, frame_count)
    tank = case_meta["tank"]
    half_width = float(tank["half_width"])
    front_y = float(layout.get("front_y", 0.04))
    bottom_z = float(tank["floor_y"])
    top_z = float(tank["top_y"])

    add_case_outline(
        name=f"{case_info['key']}_outline",
        center_x=center_x,
        half_width=half_width,
        bottom_z=bottom_z,
        top_z=top_z,
        front_y=front_y,
        material=materials["outline"],
    )
    add_text_label(
        name=f"{case_info['key']}_label",
        text=str(case_info["label"]),
        location=(center_x, 0.0, bottom_z - float(layout.get("label_offset", 0.13))),
        size=float(layout.get("label_size", 0.065)),
        material=materials["text"],
    )

    radius = float(case_meta.get("particle_display_radius", 0.005))
    points_group = get_particle_points_group(radius, materials["particles"])

    for timeline_frame, payload in enumerate(frame_payloads, start=1):
        obj = create_particle_frame_object(
            name=f"{case_info['key']}_particles_{timeline_frame:04d}",
            positions=np.asarray(payload["fluid_positions"], dtype=np.float32),
            colors=np.asarray(payload["fluid_colors"], dtype=np.float32),
            x_offset=center_x,
            axis_mode=axis_mode,
            points_group=points_group,
        )
        keyframe_visibility(obj, visible_frame=timeline_frame, frame_start=1, frame_end=frame_count)

    if "box" in case_meta:
        box_obj = create_box_object(
            name=f"{case_info['key']}_box",
            half_extent=list(case_meta["box"]["half_extent"]),
            material=materials["box"],
            axis_mode=axis_mode,
        )
        keyframe_box_animation(
            obj=box_obj,
            frame_payloads=frame_payloads,
            x_offset=center_x,
            axis_mode=axis_mode,
        )


def add_camera(total_width: float, max_top: float, layout: dict) -> None:
    target = (0.0, 0.0, 0.5 * max_top)
    bpy.ops.object.camera_add(location=(0.0, -float(layout.get("camera_y", 2.2)), 0.5 * max_top))
    camera = bpy.context.object
    camera.name = "FluidCompareCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = float(layout.get("ortho_scale", max(total_width + 0.35, max_top + 0.35)))
    cache_preview.look_at(camera, target)
    bpy.context.scene.camera = camera


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    panel_cache_dir = resolve_path(config.get("cache_dir"), args.panel_cache_dir)
    if panel_cache_dir is None:
        raise ValueError("Config must define 'cache_dir' when --panel-cache-dir is not provided.")
    panel_meta = load_json(panel_cache_dir / "panel_metadata.json")

    case_count = len(panel_meta["cases"])
    first_case_meta = load_json(Path(panel_meta["cases"][0]["path"]) / "metadata.json")
    first_tank = first_case_meta["tank"]
    case_width = 2.0 * float(first_tank["half_width"])
    case_gap = float(config.get("layout", {}).get("case_gap", 0.18))
    total_width = case_count * case_width + max(0, case_count - 1) * case_gap
    frame_count = int(args.frames or panel_meta["record_frames"])

    output_dir = resolve_path(config.get("render_dir"), args.output_dir)
    if output_dir is None:
        raise ValueError("Config must define 'render_dir' when --output-dir is not provided.")

    cache_preview.clear_scene()
    configure_scene(config, frame_count, output_dir)

    layout = config.get("layout", {})
    min_ratio = float(layout.get("colorbar_min_ratio", COLORBAR_MIN_RATIO))
    max_ratio = float(layout.get("colorbar_max_ratio", COLORBAR_MAX_RATIO))
    materials = {
        "particles": make_particle_material("FluidCompareParticles"),
        "outline": make_emission_material("FluidCompareOutline", (0.05, 0.20, 0.95, 1.0), strength=1.6),
        "text": make_emission_material("FluidCompareText", (1.0, 1.0, 1.0, 1.0), strength=1.4),
        "tick": make_emission_material("FluidCompareTick", (0.0, 0.0, 0.0, 1.0), strength=1.0),
        "box": make_emission_material("FluidCompareBox", (0.90, 0.90, 0.90, 1.0), strength=1.0),
    }
    colorbar_image = create_colorbar_image("FluidCompareColorbar", min_ratio=min_ratio, max_ratio=max_ratio)
    materials["colorbar"] = make_colorbar_material("FluidCompareColorbarMaterial", colorbar_image)

    x_positions = [
        (index - 0.5 * float(case_count - 1)) * (case_width + case_gap) for index in range(case_count)
    ]
    max_top = max(load_json(Path(case["path"]) / "metadata.json")["tank"]["top_y"] for case in panel_meta["cases"])

    add_camera(total_width, float(max_top), layout)
    add_colorbar(
        total_width=total_width,
        center_z=float(max_top) + float(layout.get("colorbar_offset", 0.16)),
        material=materials["colorbar"],
        text_material=materials["text"],
        tick_material=materials["tick"],
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    axis_mode = str(panel_meta.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    for case_info, center_x in zip(panel_meta["cases"], x_positions, strict=True):
        add_case_panel(
            case_info=case_info,
            center_x=float(center_x),
            frame_count=frame_count,
            materials=materials,
            layout=layout,
            axis_mode=axis_mode,
        )

    save_blend = None if args.skip_save_blend else resolve_path(config.get("blend_file"), args.save_blend)
    if save_blend is not None:
        save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))

    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
