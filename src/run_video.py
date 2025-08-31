# ==========================
# run_video.py (fully commented)
# ==========================
# Purpose:
#   Process a video frame-by-frame, detect + track shapes, draw overlays, and
#   export a playable video. Fixes include:
#     - Ensure EVEN frame dimensions (needed by yuv420p/H.264).
#     - Optional direct H.264 writing via imageio (--h264).
#     - Auto-fallback to imageio H.264 if OpenCV output looks unplayable.

import argparse                              # Parse command-line flags
import os                                     # Paths and file size checks
import subprocess                             # Probe ffmpeg (optional)
import cv2                                    # OpenCV: video IO + drawing
import imageio                                # ImageIO: writing H.264 reliably
from detector import detect_shapes            # Our shape detector
from depth import center_3d_from_circle, center_xy_at_Z  # Depth helpers
from tracker import CentroidTracker           # Simple centroid tracker

# ----------------------------
# Utility: ensure even width/height by padding right/bottom if needed
# ----------------------------
def ensure_even(frame):
    """
    Return a frame with even width and height by padding 1px on the right/bottom if necessary.
    Required for yuv420p/H.264 compatibility.
    """
    h, w = frame.shape[:2]                                 # Current height/width
    pad_r = 1 if (w % 2) else 0                            # Need to pad right?
    pad_b = 1 if (h % 2) else 0                            # Need to pad bottom?
    if pad_r == 0 and pad_b == 0:                          # Already even?
        return frame                                       # Return as-is
    # Pad with edge pixels (replicate) to avoid visible seams
    return cv2.copyMakeBorder(frame, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)

# ----------------------------
# Utility: safe FPS from capture
# ----------------------------
def safe_fps(cap, default=30.0):
    """Return a sane FPS value; fallback to 'default' if 0/NaN."""
    fps = cap.get(cv2.CAP_PROP_FPS)                        # Get FPS from source
    if fps is None:                                        # Missing value?
        return default                                     # Use default
    try:
        if fps <= 0 or (fps != fps):                       # Zero or NaN
            return default                                 # Use default
    except Exception:                                       # Any error
        return default                                     # Use default
    return float(fps)                                      # Cast to float

# ----------------------------
# OpenCV writers (MP4 + optional AVI)
# ----------------------------
def open_cv_writers(base_out_path, fps, width, height, also_avi=False):
    """
    Try to open MP4 (mp4v) and optionally AVI (MJPG) writers via OpenCV.
    Return (mp4_writer_or_None, mp4_path, avi_writer_or_None, avi_path_or_None).
    """
    mp4_writer = None                                      # Default: no MP4 writer
    mp4_path = base_out_path                               # MP4 path = requested --out

    try:
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')       # 'mp4v' is widely supported
        mp4_writer = cv2.VideoWriter(mp4_path, fourcc_mp4, fps, (width, height))  # Open writer
        if not mp4_writer.isOpened():                      # Failed?
            mp4_writer = None                              # Disable MP4
    except Exception:                                       # Exception creating writer
        mp4_writer = None                                  # Disable MP4

    avi_writer, avi_path = None, None                      # Defaults
    if also_avi:                                           # Only if requested
        try:
            avi_path = base_out_path.rsplit('.', 1)[0] + '.avi'  # .avi path
            fourcc_avi = cv2.VideoWriter_fourcc(*'MJPG')         # MJPG FourCC
            avi_writer = cv2.VideoWriter(avi_path, fourcc_avi, fps, (width, height))  # Open writer
            if not avi_writer.isOpened():                  # Failed?
                avi_writer, avi_path = None, None          # Disable AVI
        except Exception:                                   # Exception creating writer
            avi_writer, avi_path = None, None              # Disable AVI

    return mp4_writer, mp4_path, avi_writer, avi_path      # Return writers (or None)

# ----------------------------
# imageio-ffmpeg H.264 writer
# ----------------------------
def open_h264_writer(path, fps):
    """
    Open an imageio writer for H.264 (libx264) with yuv420p + faststart.
    Returns the 'writer' object.
    """
    return imageio.get_writer(
        path,                                              # Output path
        fps=fps,                                           # Frame rate
        codec="libx264",                                   # H.264 codec
        format="FFMPEG",                                   # Use ffmpeg backend
        output_params=[                                    # Extra ffmpeg params
            "-pix_fmt", "yuv420p",                         # Broad compatibility
            "-movflags", "+faststart"                      # Web-friendly header
        ]
    )

# ----------------------------
# Optional: do we have ffmpeg on PATH? (used for frames fallback)
# ----------------------------
def have_ffmpeg():
    """Return True if 'ffmpeg -version' runs successfully."""
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False

# ----------------------------
# Assemble frames -> MP4 with ffmpeg (fallback path)
# ----------------------------
def assemble_with_ffmpeg(frames_dir, out_mp4, fps):
    """
    Assemble PNG frames into H.264 MP4 via system ffmpeg.
    Returns True on success.
    """
    if not have_ffmpeg():                                   # ffmpeg not available
        print("[ffmpeg] Not found on PATH; cannot assemble MP4 automatically.")
        return False                                        # Bail out

    cmd = [                                                 # Build ffmpeg command
        "ffmpeg", "-y",
        "-framerate", str(int(round(fps))),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_mp4
    ]
    print("[ffmpeg] Assembling frames -> MP4 …")            # Log
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)  # Run ffmpeg
        ok = (p.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0)
        if ok:                                              # Success path
            print(f"[ffmpeg] Wrote {out_mp4} ({os.path.getsize(out_mp4)/1_000_000:.2f} MB)")
            return True
        else:                                               # Failure path
            print("[ffmpeg] Failed to assemble MP4.")
            if p.stdout:
                print("[ffmpeg stdout]", p.stdout[:500])
            if p.stderr:
                print("[ffmpeg stderr]", p.stderr[:500])
            return False
    except Exception as e:
        print(f"[ffmpeg] Exception: {e}")                   # Exception path
        return False

# ----------------------------
# Core processing
# ----------------------------
def process_video_to_outputs(inp, out_mp4, args):
    """
    Read input video, run detection+tracking per frame, draw overlays, and write
    an output that plays everywhere. Uses either:
      - OpenCV writers (default), plus auto-transcode if needed, OR
      - Direct H.264 writing via --h264 (recommended on Windows).
    """
    cap = cv2.VideoCapture(inp)                              # Open input
    if not cap.isOpened():                                   # Guard: open failure
        raise FileNotFoundError(inp)                         # Clear error

    ok, first = cap.read()                                   # Read first frame
    if not ok or first is None:                              # Guard: empty source
        cap.release()                                        # Release capture
        raise RuntimeError("Could not read first frame from input video.")

    first = ensure_even(first)                               # Ensure even W/H for H.264
    H, W = first.shape[:2]                                   # Frame dims
    fps = safe_fps(cap, default=30.0)                        # Sane FPS

    # --- Choose writer strategy ---
    h264_writer = None                                       # Placeholder for imageio writer
    mp4_writer = None                                        # Placeholder for OpenCV writer
    avi_writer = None                                        # Placeholder for AVI writer
    avi_path = None                                          # Store AVI path if created

    if args.h264:                                            # If user wants direct H.264
        h264_writer = open_h264_writer(out_mp4, fps)         # Open imageio H.264 writer
    else:
        mp4_writer, mp4_path, avi_writer, avi_path = open_cv_writers(out_mp4, fps, W, H, also_avi=args.also_avi)

    # --- Init tracker ---
    tracker = CentroidTracker(                               # Create tracker
        max_dist=args.max_dist,                              # Association distance
        max_lost=args.max_lost,                              # Drop after N lost frames
        ema_alpha=args.ema_alpha,                            # Smoothing factor
        keep_trail=args.trail                                # Trail length
    )

    # --- Plane Z for propagation (optional) ---
    propagated_Z = None                                      # Unknown until we see a circle

    # --- Helper: annotate a single frame and write it ---
    def annotate_and_write(frame):
        nonlocal propagated_Z                                # We update outer variable
        frame = ensure_even(frame)                           # Ensure even dims for safety
        annotated, dets = detect_shapes(                     # Run detector
            frame,
            K=args.K,
            min_area=args.min_area,
            solidity_min=args.solidity_min,
            use_ellipse_radius=args.ellipse_radius
        )
        tracks = tracker.update(dets)                        # Update tracks

        # If propagation requested but not yet initialized, try to lock Z from a circle
        if args.propagate_plane_z and propagated_Z is None:
            for d in dets:
                if d.get("shape") == "circle":               # Only the circle has known real size
                    (uc, vc) = d["center_px"]                # Circle center (px)
                    rc = d["radius_px"]                      # Circle radius (px)
                    xyz_c = center_3d_from_circle(uc, vc, rc, im_w=W, im_h=H)  # True (X,Y,Z)
                    if xyz_c is not None:
                        propagated_Z = xyz_c[2]              # Lock plane depth
                    break                                    # First circle is enough

        # Draw tracks + optional XYZ
        for tid, tr in tracks.items():                       # Iterate active tracks
            cx, cy = int(tr.cx), int(tr.cy)                  # Integer center for drawing

            # Find nearest detection to borrow its shape + score
            best_score, nearest_shape, bestd = None, None, 1e20
            for d in dets:
                ux, vy = d["center_px"]
                dist = (ux - cx) ** 2 + (vy - cy) ** 2
                if dist < bestd:
                    bestd = dist
                    best_score = d.get("score", None)
                    nearest_shape = d.get("shape", None)

            # Draw ID (and % score if available)
            label = f"ID {tid}" if best_score is None else f"ID {tid} ({int(round(100*max(0.0,min(1.0,best_score))))}%)"
            cv2.putText(annotated, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

            # XYZ overlay
            if args.annotate_xyz and tr.r_px > 0:
                if nearest_shape == "circle":                # True 3D only for circle
                    xyz = center_3d_from_circle(tr.cx, tr.cy, tr.r_px, im_w=W, im_h=H)
                    if xyz:
                        X, Y, Z = xyz
                        cv2.putText(annotated, f"({X:.0f},{Y:.0f},{Z:.0f}) in",
                                    (cx + 10, cy + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                else:
                    if args.propagate_plane_z and propagated_Z is not None:
                        xy = center_xy_at_Z(tr.cx, tr.cy, propagated_Z, im_w=W, im_h=H)
                        if xy is not None:
                            Xp, Yp = xy
                            cv2.putText(annotated, f"({Xp:.0f},{Yp:.0f},{propagated_Z:.0f}) in",
                                        (cx + 10, cy + 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated, "XYZ: N/A", (cx + 10, cy + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            # Motion trails (optional)
            if args.draw_trails and len(tr.trail) > 1:
                for i in range(1, len(tr.trail)):
                    cv2.line(annotated, tr.trail[i-1], tr.trail[i], (0, 200, 255), 2)

        # Write a frame using whichever writer we opened
        if h264_writer is not None:                          # Direct H.264 path
            imageio.imwrite(h264_writer, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))  # Convert BGR->RGB
        else:                                                # OpenCV writers path
            if mp4_writer is not None:
                mp4_writer.write(annotated)
            if avi_writer is not None:
                avi_writer.write(annotated)

    # --- Process first frame (already read) ---
    n = 0                                                   # Frame counter
    annotate_and_write(first)                               # Annotate+write first frame
    n += 1                                                  # Increment
    if n % 30 == 0:                                         # Progress every ~1s @30fps
        print(f"[progress] Processed {n} frames…")          # Log

    # --- Process remaining frames (respect --max_frames if given) ---
    max_frames = args.max_frames if args.max_frames > 0 else float('inf')  # Cap frames
    while n < max_frames:
        ok, frame = cap.read()                              # Read next frame
        if not ok or frame is None:                         # EOF or read error
            break                                           # Exit loop
        annotate_and_write(frame)                           # Annotate+write frame
        n += 1                                              # Increment
        if n % 30 == 0:                                     # Progress log
            print(f"[progress] Processed {n} frames…")      # Log

    # --- Cleanup writers/capture ---
    cap.release()                                           # Close input capture
    if h264_writer is not None:                             # If using imageio
        h264_writer.close()                                 # Close writer
    if mp4_writer is not None:                              # If using OpenCV MP4
        mp4_writer.release()                                # Release writer
    if avi_writer is not None:                              # If using AVI
        avi_writer.release()                                # Release writer

    # --- If we used OpenCV writers, check file size and auto-transcode if tiny/unplayable ---
    if h264_writer is None:                                 # Only applies to OpenCV path
        def size_or_zero(p):                                # Helper: safe size
            try:
                return os.path.getsize(p) if p and os.path.exists(p) else 0
            except Exception:
                return 0

        mp4_size = size_or_zero(out_mp4)                    # Output MP4 size
        avi_size = size_or_zero(out_mp4.rsplit('.', 1)[0] + '.avi') if args.also_avi else 0  # AVI size if any
        print(f"[done] frames: {n}  MP4: {mp4_size/1_000_000:.2f} MB  AVI: {avi_size/1_000_000:.2f} MB")

        if mp4_size < 20_000 and avi_size < 20_000:         # Both tiny => likely unplayable
            print("[fallback] OpenCV outputs seem unplayable; transcoding to H.264 with imageio…")
            # Transcode: read OpenCV MP4 back in and write a clean H.264 MP4
            try:
                # Open a new reader to the (possibly broken) file; if it fails,
                # we can still assemble from frames fallback below.
                cap2 = cv2.VideoCapture(out_mp4)
                ok2, fr2 = cap2.read()
                if ok2 and fr2 is not None:
                    fr2 = ensure_even(fr2)
                    H2, W2 = fr2.shape[:2]
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Create a clean H.264 writer next to the original name
                    clean_path = out_mp4.rsplit('.', 1)[0] + "_h264.mp4"
                    wr = open_h264_writer(clean_path, fps)
                    count = 0
                    while True:
                        okf, f = cap2.read()
                        if not okf or f is None:
                            break
                        f = ensure_even(f)
                        imageio.imwrite(wr, cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                        count += 1
                        if count % 60 == 0:
                            print(f"[h264] wrote {count} frames…")
                    wr.close()
                    cap2.release()
                    print(f"[h264] Wrote {clean_path} — please try this file.")
                else:
                    cap2.release()
                    print("[fallback] Could not re-open MP4 for transcode; consider frames+ffmpeg route.")
            except Exception as e:
                print(f"[h264] Transcode exception: {e}")
    else:
        # If we used imageio H.264 directly, just report success (file will be playable).
        print(f"[done] frames: {n}  MP4: {out_mp4}")

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PennAiR: video detection + tracking + (X,Y,Z) with robust H.264 export"
    )
    parser.add_argument("--in",  dest="inp",  required=True, help="Input video path")     # Required input
    parser.add_argument("--out", dest="outp", required=True, help="Output MP4 path")      # Required output

    # Detector params
    parser.add_argument("--K", type=int, default=5, help="k-means clusters (default 5)")
    parser.add_argument("--min_area", type=int, default=300, help="min component area (px^2)")
    parser.add_argument("--solidity_min", type=float, default=0.85, help="min solidity [0..1]")
    parser.add_argument("--ellipse_radius", action="store_true", help="use ellipse minor axis for radius")

    # Tracker params
    parser.add_argument("--max_dist", type=float, default=60.0, help="max association distance (px)")
    parser.add_argument("--max_lost", type=int,   default=10,   help="max frames to keep unmatched")
    parser.add_argument("--ema_alpha", type=float, default=0.5, help="EMA smoothing factor 0..1")
    parser.add_argument("--trail", type=int, default=20, help="trail length for drawing")

    # Output / behavior
    parser.add_argument("--annotate_xyz", action="store_true", help="draw (X,Y,Z) overlay")
    parser.add_argument("--draw_trails", action="store_true", help="draw motion trails")
    parser.add_argument("--also_avi", action="store_true", help="also export AVI via OpenCV")
    parser.add_argument("--max_frames", type=int, default=-1, help="cap frames processed; -1=all")
    parser.add_argument("--propagate_plane_z", action="store_true",
                        help="use circle-derived Z for all shapes (flat-surface approx)")
    parser.add_argument("--h264", action="store_true",
                        help="write H.264 directly with imageio (recommended on Windows)")

    args = parser.parse_args()                             # Parse arguments
    process_video_to_outputs(args.inp, args.outp, args)    # Run pipeline

if __name__ == "__main__":                                 # Entry-point guard
    main()                                                 # Execute
