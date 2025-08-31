@'
# PennAiR Shapes — Vision Challenge (Fall 2025)

Detect solid shapes on a grassy (or arbitrary) background, trace their outlines, locate centers, compute 3-D for the circle, and run it on video. This repo implements **Parts 1–4** plus the **extra-credit background/gradient robustness**. (ROS2/Part 5 intentionally omitted per submission scope.)

---

## TL;DR — How to run (Windows / PowerShell)

> From the project root where `src/` and `data/` live.

```powershell
# 0) Create & activate venv, then install requirements
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install imageio imageio-ffmpeg   # H.264 writer for universally playable MP4s

# 1) Static image (just circle gets real XYZ)
python src\run_static.py --in data\PennAir_2024_App_Static.png `
  --out outputs\static_annotated.png --annotate_xyz --ellipse_radius

# 2) Static image with plane-Z propagation (all shapes get XYZ under flat-ground assumption)
python src\run_static.py --in data\PennAir_2024_App_Static.png `
  --out outputs\static_annotated_plane.png --annotate_xyz --ellipse_radius --propagate_plane_z

# 3) Dynamic video (recommended: write H.264 directly)
python src\run_video.py --in data\PennAir_2024_App_Dynamic.mp4 `
  --out outputs\dynamic_annotated_h264.mp4 --annotate_xyz --draw_trails `
  --ellipse_radius --propagate_plane_z

# 4) Hard video (extra-credit proof)
python src\run_video.py --in data\PennAir_2024_App_Dynamic_Hard.mp4 `
  --out outputs\dynamic_hard_annotated_h264.mp4 --annotate_xyz --draw_trails `
  --ellipse_radius --propagate_plane_z
