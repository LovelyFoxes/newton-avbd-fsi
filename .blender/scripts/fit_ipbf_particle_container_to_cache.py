from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


GLASS_NAMES = ("GlassLeft", "GlassRight", "GlassBack", "GlassFront")
WIREFRAME_NAME = "IPBFContainerWireframe"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Fit IPBF particle render containers to their cached scene metadata.")
    parser.add_argument("--blend", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(".blender/cache/ipbf_particle_examples"))
    parser.add_argument("--wall-thickness", type=float, default=0.01)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def remove_object(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def disable_shadow(obj: bpy.types.Object) -> None:
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False


def material_or_default(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    axis_mode: str,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=to_blender_position(location, axis_mode))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = to_blender_scale(scale, axis_mode)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    disable_shadow(obj)


def add_glass(metadata: dict, wall_thickness: float) -> None:
    tank = metadata.get("tank", {})
    axis_mode = str(metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    half_width = float(tank.get("half_width", 1.0))
    half_depth = float(tank.get("half_depth", 1.0))
    floor_y = float(tank.get("floor_y", 0.0))
    top_y = float(tank.get("top_y", 1.0))
    height = max(top_y - floor_y, 1.0e-5)
    center_y = floor_y + 0.5 * height

    glass = material_or_default("CleanGlass", (0.96, 0.99, 1.0, 0.0))
    panes = [
        ("GlassLeft", (-(half_width + 0.5 * wall_thickness), center_y, 0.0), (wall_thickness, height, 2.0 * half_depth)),
        ("GlassRight", ((half_width + 0.5 * wall_thickness), center_y, 0.0), (wall_thickness, height, 2.0 * half_depth)),
        ("GlassBack", (0.0, center_y, -(half_depth + 0.5 * wall_thickness)), (2.0 * half_width, height, wall_thickness)),
        ("GlassFront", (0.0, center_y, (half_depth + 0.5 * wall_thickness)), (2.0 * half_width, height, wall_thickness)),
    ]
    for name, location, scale in panes:
        add_box(name, location, scale, glass, axis_mode)


def add_wireframe(metadata: dict) -> None:
    starts = np.asarray(metadata.get("wireframe_starts", []), dtype=np.float32)
    ends = np.asarray(metadata.get("wireframe_ends", []), dtype=np.float32)
    if starts.size == 0 or ends.size == 0 or starts.shape != ends.shape:
        return

    axis_mode = str(metadata.get("axis_conversion", "newton-y-up-to-blender-z-up"))
    starts_b = to_blender_positions(starts, axis_mode)
    ends_b = to_blender_positions(ends, axis_mode)
    material = material_or_default("GlassEdges", (0.0, 0.0, 0.0, 0.92))

    curve_data = bpy.data.curves.new(WIREFRAME_NAME, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.0015
    curve_data.fill_mode = "FULL"
    for start, end in zip(starts_b, ends_b, strict=True):
        spline = curve_data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (float(start[0]), float(start[1]), float(start[2]), 1.0)
        spline.points[1].co = (float(end[0]), float(end[1]), float(end[2]), 1.0)
    obj = bpy.data.objects.new(WIREFRAME_NAME, curve_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    disable_shadow(obj)


def fit_blend(blend_path: Path, cache_root: Path, wall_thickness: float) -> None:
    case_key = blend_path.stem
    metadata_path = cache_root / case_key / "metadata.json"
    metadata = load_json(metadata_path)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    for name in (*GLASS_NAMES, WIREFRAME_NAME):
        remove_object(name)
    add_glass(metadata, wall_thickness)
    add_wireframe(metadata)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    tank = metadata.get("tank", {})
    print(
        "[IPBF Container Fit] "
        f"{blend_path.name}: half_width={tank.get('half_width')}, "
        f"half_depth={tank.get('half_depth')}, top_y={tank.get('top_y')}"
    )


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    for blend in args.blend:
        fit_blend(blend.resolve(), cache_root, args.wall_thickness)


if __name__ == "__main__":
    main()
