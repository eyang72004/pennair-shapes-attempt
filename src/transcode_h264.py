# ==========================
# transcode_h264.py (fully commented)
# ==========================
# Converts any input video into a universally playable H.264 MP4 using
# imageio-ffmpeg (bundled ffmpeg). This avoids OS codec issues.
# Usage:
#   python src/transcode_h264.py --in outputs/dynamic_annotated_test.mp4 --out outputs/dynamic_annotated_test_h264.mp4

import argparse                 # For parsing command-line flags
import imageio                  # ImageIO provides a simple video writer API
import imageio.v3 as iio        # v3 API (we won’t use here but good to import)
import cv2                      # OpenCV to read frames from the source video
import os                       # For file checks and sizes
import numpy as np              # For array handling
                              
def main():
    # Build the CLI
    parser = argparse.ArgumentParser(description="Transcode any video to H.264 MP4 (yuv420p, faststart)")
    parser.add_argument("--in",  dest="inp",  required=True, help="Input video path")
    parser.add_argument("--out", dest="outp", required=True, help="Output H.264 MP4 path")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: read from input or 30)")
    args = parser.parse_args()

    # Open the input video with OpenCV
    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise FileNotFoundError(args.inp)

    # Try to read FPS from source, fall back to 30 if missing
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if args.fps is not None:
        fps = float(args.fps)
    else:
        fps = float(src_fps) if src_fps and src_fps > 0 and src_fps == src_fps else 30.0  # handle NaN

    # Read first frame to lock width/height and validate input
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    # Determine frame size
    h, w = frame_bgr.shape[:2]

    # Create the H.264 writer via imageio-ffmpeg.
    # We set:
    #   - codec='libx264'      (H.264)
    #   - pix_fmt=yuv420p      (widely supported)
    #   - movflags=+faststart  (web-friendly MP4 header)
    # Note: imageio picks the ffmpeg binary provided by imageio-ffmpeg.
    writer = imageio.get_writer(
        args.outp,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )

    # Write first frame (convert BGR->RGB for imageio/ffmpeg)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)

    # Write the rest of the frames
    n = 1
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        n += 1
        if n % 60 == 0:
            print(f"[transcode] wrote {n} frames…")

    # Clean up
    writer.close()
    cap.release()

    # Report file size
    try:
        sz = os.path.getsize(args.outp) / 1_000_000.0
        print(f"[done] Wrote {args.outp} ({sz:.2f} MB) at {fps:.2f} fps, frames: {n}")
    except Exception:
        print(f"[done] Wrote {args.outp}")

if __name__ == "__main__":
    main()
