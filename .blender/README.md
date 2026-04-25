# Blender FSI Render Workspace

This folder is a local, git-ignored workspace for turning Newton FSI simulation
results into offline Blender renders. Keep large caches, `.blend` files, image
sequences, and temporary exports here.

## Folder Layout

```text
.blender/
  cache/       Newton-exported frame caches, usually metadata.json + frames/*.npz
  config/      Render profiles and scene defaults
  renders/     Rendered image sequences and videos
  scripts/     Blender Python helper scripts
  templates/   Generated .blend scene templates
  usd/         Optional USD previews exported from Newton
```

## Recommended Pipeline

```text
Newton simulation
-> export particle/body cache
-> Blender imports cache
-> create tank, water material, sphere buoys, camera, lights
-> render fixed 30-frame image sequence
```

For now, the Blender script renders a preview-style particle water pass by
instancing a limited number of small translucent water spheres. This is useful
for composition, lighting, and debugging. For final thesis-quality water, the
next step is to convert the cached fluid particles into a surface mesh using
Blender Geometry Nodes, OpenVDB, Houdini Particle Fluid Surface, or an external
tool such as splashsurf.

## Export A Newton USD Preview

Newton can already write a time-sampled USD preview without opening the realtime
OpenGL window:

```powershell
uv run -m newton.examples fsi_ipbf_vbd_three_sphere_buoys `
  --viewer usd `
  --output-path .blender/usd/fsi_three_sphere_buoys_preview.usd `
  --device cuda:0 `
  --num-frames 30 `
  --quiet
```

This USD preview is useful for checking transforms and point positions in
Omniverse or Blender, but the fluid is still point data rather than a real water
surface.

## Export A Newton Simulation Cache

The tracked exporter is independent from Newton examples and viewer code. It
builds the scene, advances `SolverFSI` headlessly, writes simulation frames, and
records Newton simulation timing separately from cache I/O:

```powershell
uv run python .blender/scripts/export_fsi_cache.py `
  --scene three_sphere_buoys `
  --device cuda:0 `
  --output-dir .blender/cache/fsi_three_sphere_buoys `
  --warmup-frames 60 `
  --record-frames 30 `
  --record-stride 1 `
  --overwrite
```

The `metadata.json` file contains title-card-ready lines such as particle count,
rigid body count, timestep, iteration counts, and `avg_sim_ms_per_frame`. That
simulation time is measured around Newton solver advancement only, with Warp
device synchronization before and after each engine frame. It excludes viewer,
Blender rendering, one-time Warp module load, and cache writing.

## One-Click Windows Batch

For daily use on this machine, run:

```powershell
.blender\run_fsi_render.bat
```

The batch file is local and git-ignored. Edit the variable block at the top to
switch coupling method, frame count, cache/render folders, preview particle
count, or whether to export cache, save a `.blend`, or render images.

Useful toggles:

```bat
set "EXPORT_CACHE=1"      rem 1 = rerun Newton, 0 = reuse existing cache
set "RENDER_IMAGES=1"    rem 1 = render PNG sequence, 0 = skip image render
set "SAVE_BLEND=0"       rem 1 = save a .blend file for manual editing
set "START_TIME_SECONDS=0.0"
set "DURATION_SECONDS=10.0"
set "VIDEO_FPS=30"
set "COUPLING_MODE=interlinked"
set "RECORD_INITIAL_FRAME=1"
set "RENDER_FRAMES=auto"
set "WATER_MAX_PARTICLES=2500"
```

The batch file defaults to time-based export. For example:

```bat
set "START_TIME_SECONDS=0.0"
set "DURATION_SECONDS=10.0"
set "VIDEO_FPS=30"
```

exports 300 cached render frames starting at simulation time 0.0 s. Internally,
the exporter derives the engine-frame warmup and record stride from these values.
The metadata records `frame_dt`, `sim_dt`, start time, video FPS, stride, and
recorded duration, so the rendered video and thesis caption can stay consistent.

The Blender importer applies a right-handed Y-up to Z-up conversion by default:

```text
(x, y, z)_Newton -> (x, -z, y)_Blender
```

Override it with `--axis-conversion none` in `EXTRA_RENDER_ARGS` only if you want
to inspect raw Newton coordinates inside Blender.

## Splashsurf Meshing

For thesis-quality water, use splashsurf on the exported fluid cache instead of
rendering preview water particles directly.

Run the local helper:

```powershell
.blender\run_splashsurf_meshing.bat
```

It performs two steps:

```text
cache frame_*.npz
-> particles/frame_*.ply
-> pysplashsurf reconstruct
-> surface_obj/frame_*.obj
```

The particle export helper writes:

```text
.blender/cache/.../particles/splashsurf_particles_metadata.json
```

This file stores the recommended splashsurf parameters derived from the Newton
cache metadata, including particle radius and smoothing length.

The default particle export also converts coordinates to Blender space before
writing PLY files, so the resulting splashsurf meshes can be imported into
Blender without another Y-up/Z-up correction.

## Splashsurf Surface Scene

To build a `.blend` scene that uses the splashsurf OBJ sequence instead of
preview water particles, run:

```powershell
.blender\run_fsi_surface_render.bat
```

This uses:

```text
.blender/scripts/render_fsi_surface_sequence.py
```

and builds a scene with:

- Tank geometry
- Animated buoy spheres
- Splashsurf OBJ mesh sequence as the visible water surface
- No preview particle water

This path is heavier than the preview path because it imports one mesh object per
frame and switches visibility over time, but it is the current built-in route to
generate a true continuous water surface `.blend` without relying on extra
Blender add-ons.

## Render From A Newton Cache

The exporter writes the cache like this:

```text
.blender/cache/fsi_three_sphere_buoys/
  metadata.json
  frames/
    frame_0000.npz
    frame_0001.npz
    ...
```

Then render a 30-frame preview:

```powershell
blender --background --python .blender/scripts/render_fsi_cache.py -- `
  --cache-dir .blender/cache/fsi_three_sphere_buoys `
  --output-dir .blender/renders/fsi_three_sphere_buoys `
  --frames 30 `
  --water-max-particles 2500 `
  --render
```

If no cache is available yet, the same script can still create a template scene:

```powershell
blender --background --python .blender/scripts/render_fsi_cache.py -- `
  --output-dir .blender/renders/template `
  --save-blend .blender/templates/fsi_three_sphere_buoys_template.blend
```

## Cache Schema Target

The render script is intentionally tolerant, but the preferred cache keys are:

```text
fluid_positions: float32 [N, 3]
fluid_radii:     float32 [N]
body_q:          float32 [B, 7], Newton transform order [x, y, z, qx, qy, qz, qw]
body_labels:     optional list[str] in metadata.json
sphere_radii:    optional float32 [B] in metadata.json or frame file
sphere_densities: optional float32 [B] in metadata.json
tank_half_extents: optional [half_width, half_height, half_depth]
```

For final rendering, do not render FSI boundary samples as the visible rigid
body. Boundary samples are solver/debug data. Render analytic sphere meshes from
the cached rigid transforms instead.
