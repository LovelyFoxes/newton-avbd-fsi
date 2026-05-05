from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


BASE_STYLE_OBJECTS = {
    "IPBFParticleCamera",
    "CleanFloor",
    "GlassFront",
    "GlassBack",
    "GlassLeft",
    "GlassRight",
    "IPBFContainerWireframe",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Copy the hand-tuned IPBF particle render style between .blend files.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    return parser.parse_args(argv)


def purge_unused() -> None:
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def source_style_names(source: Path) -> list[str]:
    bpy.ops.wm.open_mainfile(filepath=str(source))
    names: list[str] = []
    for obj in bpy.data.objects:
        if obj.name in BASE_STYLE_OBJECTS or obj.type == "LIGHT":
            names.append(obj.name)
    return sorted(set(names))


def remove_target_style_objects(names: list[str]) -> None:
    remove_names = set(names)
    for obj in list(bpy.data.objects):
        if obj.name in remove_names or obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    purge_unused()


def append_style_objects(source: Path, names: list[str]) -> list[bpy.types.Object]:
    with bpy.data.libraries.load(str(source), link=False) as (data_from, data_to):
        available = set(data_from.objects)
        data_to.objects = [name for name in names if name in available]

    appended: list[bpy.types.Object] = []
    for obj in data_to.objects:
        if obj is None:
            continue
        bpy.context.scene.collection.objects.link(obj)
        appended.append(obj)
    return appended


def apply_to_target(source: Path, target: Path, names: list[str]) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(target))
    remove_target_style_objects(names)
    appended = append_style_objects(source, names)
    camera = next((obj for obj in appended if obj.type == "CAMERA"), None)
    if camera is not None:
        bpy.context.scene.camera = camera
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    print(f"[IPBF Style Sync] Updated {target}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    targets = [target.resolve() for target in args.target]
    names = source_style_names(source)
    if not names:
        raise RuntimeError(f"No style objects found in {source}")
    print(f"[IPBF Style Sync] Source objects: {', '.join(names)}")
    for target in targets:
        if target == source:
            continue
        if not target.exists():
            raise FileNotFoundError(target)
        apply_to_target(source, target, names)


if __name__ == "__main__":
    main()
