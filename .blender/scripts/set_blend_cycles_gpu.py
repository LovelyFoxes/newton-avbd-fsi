from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Set existing .blend files to Cycles GPU rendering.")
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--compute-device-type", type=str, default="OPTIX")
    parser.add_argument("--samples", type=int, default=96)
    return parser.parse_args(argv)


def configure_cycles_gpu(compute_device_type: str) -> str:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    preferences = bpy.context.preferences.addons.get("cycles")
    if preferences is None:
        return "UNKNOWN"

    cycles_preferences = preferences.preferences
    preferred = [compute_device_type.upper(), "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]
    for device_type in dict.fromkeys(preferred):
        try:
            cycles_preferences.compute_device_type = device_type
            cycles_preferences.get_devices()
        except Exception:
            continue
        devices = list(getattr(cycles_preferences, "devices", []))
        gpu_devices = [device for device in devices if getattr(device, "type", "") != "CPU"]
        if not gpu_devices:
            continue
        for device in devices:
            device.use = getattr(device, "type", "") != "CPU"
        return device_type
    return "CPU_FALLBACK"


def main() -> None:
    args = parse_args()
    for target in args.target:
        path = target.resolve()
        bpy.ops.wm.open_mainfile(filepath=str(path))
        selected = configure_cycles_gpu(args.compute_device_type)
        bpy.context.scene.cycles.samples = args.samples
        bpy.context.scene.cycles.use_denoising = True
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        print(f"[Cycles GPU] {path} -> {selected}")


if __name__ == "__main__":
    main()
