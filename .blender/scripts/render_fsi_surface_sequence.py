"""Build a Blender scene from a splashsurf OBJ mesh sequence plus Newton rigid caches.

Run from Blender, for example:

    blender --background --python .blender/scripts/render_fsi_surface_sequence.py -- \
      --cache-dir .blender/cache/fsi_three_sphere_buoys \
      --surface-dir .blender/cache/fsi_three_sphere_buoys/surface_obj \
      --output-dir .blender/renders/fsi_three_sphere_buoys_surface \
      --save-blend .blender/templates/fsi_three_sphere_buoys_surface.blend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing metadata.json and frames/frame_*.npz.",
    )
    parser.add_argument(
        "--surface-dir",
        type=Path,
        default=None,
        help="Directory containing splashsurf OBJ sequence. Defaults to <cache-dir>/surface_obj.",
    )
    parser.add_argument(
        "--particles-dir",
        type=Path,
        default=None,
        help="Directory containing splashsurf_particles_metadata.json. Defaults to <cache-dir>/particles.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".blender/renders/fsi_surface"))
    parser.add_argument("--config", type=Path, default=Path(".blender/config/three_sphere_buoys_render.json"))
    parser.add_argument("--save-blend", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Number of splashsurf frames to import.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--axis-conversion",
        choices=["newton-y-up-to-blender-z-up", "none"],
        default=None,
        help="Coordinate conversion applied to Newton rigid cache data.",
    )
    parser.add_argument("--render", action="store_true", help="Render an image sequence after scene setup.")
    return parser.parse_args(argv)


def load_surface_paths(surface_dir: Path) -> list[Path]:
    if not surface_dir.exists():
        raise FileNotFoundError(f"Surface directory does not exist: {surface_dir}")
    return sorted(surface_dir.glob("frame_*.obj"))


def load_particles_metadata(particles_dir: Path | None) -> dict:
    if particles_dir is None:
        return {}
    return cache_preview.load_json(particles_dir / "splashsurf_particles_metadata.json")


def build_surface_frame_records(
    selected_surface_paths: list[Path],
    particles_metadata: dict,
    cache_dir: Path,
) -> list[dict]:
    export_frames = particles_metadata.get("frames", [])
    export_map = {int(entry["export_index"]): entry for entry in export_frames}
    records: list[dict] = []

    for path in selected_surface_paths:
        export_index = parse_frame_index(path)
        export_entry = export_map.get(export_index)
        if export_entry is not None:
            cache_frame_path = Path(export_entry["source_path"])
            sim_time = export_entry.get("sim_time")
            source_frame = int(export_entry["source_frame"])
        else:
            cache_frame_path = cache_dir / "frames" / f"frame_{export_index:04d}.npz"
            sim_time = None
            source_frame = export_index
        records.append(
            {
                "surface_path": path,
                "export_index": export_index,
                "cache_frame_path": cache_frame_path,
                "source_frame": source_frame,
                "sim_time": sim_time,
            }
        )
    return records


def parse_frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def build_scene_materials(config: dict) -> dict[str, bpy.types.Material]:
    materials = cache_preview.build_materials(config)
    surface_material = cache_preview.make_material(
        "Water_Surface",
        (0.67, 0.84, 1.0, 1.0),
        roughness=0.03,
        metallic=0.0,
        alpha_blend=False,
        transmission=1.0,
    )

    if surface_material.use_nodes:
        nodes = surface_material.node_tree.nodes
        links = surface_material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        output = nodes.get("Material Output")
        if bsdf is not None:
            cache_preview.set_node_input(bsdf, "IOR", 1.333)
            cache_preview.set_node_input(bsdf, "Base Color", (0.63, 0.84, 1.0, 1.0))
            cache_preview.set_node_input(bsdf, "Roughness", 0.025)
        if output is not None:
            volume = nodes.new("ShaderNodeVolumeAbsorption")
            volume.name = "Water_Volume_Absorption"
            cache_preview.set_node_input(volume, "Color", (0.12, 0.35, 0.95, 1.0))
            cache_preview.set_node_input(volume, "Density", 0.15)
            links.new(volume.outputs["Volume"], output.inputs["Volume"])

    materials["water_surface"] = surface_material
    return materials


def create_surface_collection(name: str = "Water_Surface_Sequence") -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def import_surface_frame(path: Path, collection: bpy.types.Collection, material: bpy.types.Material) -> list[bpy.types.Object]:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z", validate_meshes=True)
    imported = list(bpy.context.selected_objects)
    for obj in imported:
        for existing_collection in list(obj.users_collection):
            existing_collection.objects.unlink(obj)
        collection.objects.link(obj)
        assign_material(obj, material)
    return imported


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


def import_surface_sequence(
    records: list[dict],
    collection: bpy.types.Collection,
    material: bpy.types.Material,
    *,
    frame_start: int,
    frame_end: int,
) -> None:
    for timeline_frame, record in enumerate(records, start=frame_start):
        imported = import_surface_frame(record["surface_path"], collection, material)
        for index, obj in enumerate(imported):
            obj.name = f"WaterSurface_{timeline_frame:04d}_{index:02d}"
            keyframe_visibility(obj, visible_frame=timeline_frame, frame_start=frame_start, frame_end=frame_end)


def load_cache_frames(records: list[dict]) -> list[dict]:
    frames: list[dict] = []
    for record in records:
        frames.append(cache_preview.read_frame(record["cache_frame_path"]))
    return frames


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    surface_dir = args.surface_dir if args.surface_dir is not None else cache_dir / "surface_obj"
    particles_dir = args.particles_dir if args.particles_dir is not None else cache_dir / "particles"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = cache_preview.load_json(args.config)
    metadata = cache_preview.load_json(cache_dir / "metadata.json")
    particles_metadata = load_particles_metadata(particles_dir)
    axis_mode = cache_preview.resolve_axis_mode(args, config, metadata)

    selected_surface_paths = cache_preview.selected_frames(
        load_surface_paths(surface_dir),
        args.start_frame,
        args.stride,
        args.frames,
    )
    if not selected_surface_paths:
        raise FileNotFoundError(f"No splashsurf OBJ sequence frames found in {surface_dir}")

    records = build_surface_frame_records(selected_surface_paths, particles_metadata, cache_dir)
    cache_frames = load_cache_frames(records)

    cache_preview.clear_scene()
    cache_preview.configure_render(config, metadata, output_dir)
    materials = build_scene_materials(config)
    cache_preview.add_tank(config, metadata, materials, axis_mode)
    cache_preview.add_camera_and_lights(config)
    cache_preview.add_buoys(cache_frames, metadata, config, materials, axis_mode)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(records)

    surface_collection = create_surface_collection()
    import_surface_sequence(
        records,
        surface_collection,
        materials["water_surface"],
        frame_start=scene.frame_start,
        frame_end=scene.frame_end,
    )

    if args.save_blend is not None:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))

    if args.render:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
