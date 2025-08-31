# ==========================
# run_static.py (fully commented)
# ==========================
# Purpose:
#   Run detection on a single image. Optionally overlay (X,Y,Z) in inches.
#   Modes:
#     - Default (conservative): show XYZ ONLY for the circle (known real size).
#     - --propagate_plane_z: compute Z from the circle once and reuse that Z
#       for ALL shapes (assumes flat surface at roughly constant distance).

import argparse                              # Parse command-line flags
import cv2                                   # Image I/O and drawing
from detector import detect_shapes           # Detection (contours, centers, shape labels)
from depth import (                          # Depth helpers
    center_3d_from_circle,                   # (X,Y,Z) from circle center+radius
    center_xy_at_Z                           # (X,Y) at chosen Z for any pixel
)

def main():                                   # Entry point
    parser = argparse.ArgumentParser(
        description="PennAiR: static detection (+ optional XYZ; plane propagation optional)"
    )
    parser.add_argument("--in",  dest="inp",  required=True, help="Input image path")      # Input path
    parser.add_argument("--out", dest="outp", required=True, help="Output annotated path") # Output path
    parser.add_argument("--K", type=int, default=5, help="k-means clusters (default 5)")   # Clusters
    parser.add_argument("--min_area", type=int, default=300, help="min component area")    # Area gate
    parser.add_argument("--solidity_min", type=float, default=0.85, help="min solidity")   # Solidity gate
    parser.add_argument("--ellipse_radius", action="store_true", help="use ellipse minor axis as radius")  # Ellipse fit
    parser.add_argument("--annotate_xyz", action="store_true", help="draw (X,Y,Z) overlay")                # Toggle XYZ
    parser.add_argument("--propagate_plane_z", action="store_true",
                        help="use circle depth as plane Z for ALL shapes (flat-surface assumption)")       # New flag
    args = parser.parse_args()                    # Parse flags

    img = cv2.imread(args.inp)                    # Load image from disk
    if img is None:                               # Guard for missing file
        raise FileNotFoundError(f"Could not load: {args.inp}")  # Clear error

    annotated, dets = detect_shapes(              # Run shape detector
        img,
        K=args.K,
        min_area=args.min_area,
        solidity_min=args.solidity_min,
        use_ellipse_radius=args.ellipse_radius
    )

    if args.annotate_xyz:                         # If XYZ overlay was requested
        H, W = annotated.shape[:2]                # Image dimensions (for principal point at image center)

        # --- Optionally compute plane Z from the circle once, then reuse ---
        plane_Z = None                            # Initialize no plane depth known yet
        if args.propagate_plane_z:                # Only if user enabled propagation
            for d in dets:                        # Search for a circle among detections
                if d.get("shape") == "circle":    # Found the circle (has known physical size)
                    (uc, vc) = d["center_px"]     # Circle center in pixels
                    rc = d["radius_px"]           # Circle radius in pixels
                    xyz_c = center_3d_from_circle(uc, vc, rc, im_w=W, im_h=H)  # Compute its (X,Y,Z)
                    if xyz_c is not None:         # If successful
                        plane_Z = xyz_c[2]        # Use the circle's Z as the plane depth
                    break                         # Only need the first circle

        # --- Annotate each detection ---
        for d in dets:                            # Loop through detections
            (u, v) = d["center_px"]               # Pixel center
            if d.get("shape") == "circle":        # Circle: compute real (X,Y,Z)
                r = d["radius_px"]                # Pixel radius for the circle
                xyz = center_3d_from_circle(u, v, r, im_w=W, im_h=H)  # True (X,Y,Z) using known radius
                if xyz:                           # If we got a valid result
                    X, Y, Z = xyz                 # Unpack
                    cv2.putText(annotated, f"({X:.1f},{Y:.1f},{Z:.1f}) in",  # Draw real XYZ
                                (u + 8, v + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                if args.propagate_plane_z and plane_Z is not None:     # If we chose to propagate and have Z
                    xy = center_xy_at_Z(u, v, plane_Z, im_w=W, im_h=H) # Compute (X,Y) at common plane Z
                    if xy is not None:                                 # If successful
                        Xp, Yp = xy                                     # Unpack
                        cv2.putText(annotated, f"({Xp:.1f},{Yp:.1f},{plane_Z:.1f}) in",  # Approx XYZ
                                    (u + 8, v + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
                    else:                                               # If XY failed (very rare)
                        cv2.putText(annotated, "XYZ: N/A",              # Be honest: failed computation
                                    (u + 8, v + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                else:                                                   # Conservative: don't propagate
                    cv2.putText(annotated, "XYZ: N/A",                  # No physical size → no Z
                                (u + 8, v + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    ok = cv2.imwrite(args.outp, annotated)        # Save output image
    if not ok:                                    # Guard: write failure
        raise RuntimeError(f"Failed to write {args.outp}")             # Clear error
    print(f"Wrote {args.outp} with {len(dets)} detections.")           # Console summary

if __name__ == "__main__":                        # Script guard
    main()                                        # Run the tool
