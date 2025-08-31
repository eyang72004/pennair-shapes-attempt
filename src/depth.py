# ==========================
# depth.py (fully commented)
# ==========================
# Purpose:
#   Convert a 2D pixel center (u,v) and its pixel radius r_px (for a known-size
#   circle) into 3D coordinates (X,Y,Z) in inches using a pinhole camera model.
#   Also provide a helper to compute (X,Y) at a chosen plane depth Z for any pixel.

from typing import Optional, Tuple  # Optional=may return None; Tuple=typed tuples  # noqa: E702

# --- Camera intrinsics (given by the challenge) ---  # noqa: E265
FX = 2564.3186869      # Focal length along x axis in pixels
FY = 2569.70273111     # Focal length along y axis in pixels
CX = 0.0               # Principal point x (as provided; top-left origin)
CY = 0.0               # Principal point y (as provided; top-left origin)

# --- Known real-world circle radius ---  # noqa: E265
R_IN = 10.0            # Real radius of the marker circle, in inches

def center_3d_from_circle(
    u_px: float,        # Pixel x coordinate of the circle center
    v_px: float,        # Pixel y coordinate of the circle center
    r_px: float,        # Pixel radius of the circle in the image
    im_w: float = None, # Optional: image width (if given, use image center as cx)
    im_h: float = None  # Optional: image height (if given, use image center as cy)
) -> Optional[Tuple[float, float, float]]:
    """
    Compute (X, Y, Z) in inches for the **circle** only.

    Model (similar triangles):
      r_px ≈ f * R / Z   =>   Z ≈ f * R / r_px
      X = (u - cx) * Z / FX
      Y = (v - cy) * Z / FY

    If im_w/im_h are provided, we set the principal point (cx,cy) to the image center;
    otherwise we use (CX, CY) as defined above (top-left origin).
    """
    if r_px <= 0:                                   # Guard against invalid radius
        return None                                 # Cannot compute depth → return None
    Z = FX * R_IN / r_px                            # Depth in inches (using FX)
    cx = (im_w / 2.0) if im_w is not None else CX   # Principal point x: prefer image center
    cy = (im_h / 2.0) if im_h is not None else CY   # Principal point y: prefer image center
    X = (u_px - cx) * Z / FX                        # Back-project pixel u to metric X (inches)
    Y = (v_px - cy) * Z / FY                        # Back-project pixel v to metric Y (inches)
    return (float(X), float(Y), float(Z))           # Return floats for consistency

def center_xy_at_Z(
    u_px: float,        # Pixel x coordinate
    v_px: float,        # Pixel y coordinate
    Z_in: float,        # Chosen plane depth in inches
    im_w: float = None, # Optional: image width (for image-center principal point)
    im_h: float = None  # Optional: image height (for image-center principal point)
) -> Optional[Tuple[float, float]]:
    """
    Back-project pixel (u,v) to metric (X,Y) given a chosen depth Z_in.
    Useful when assuming a flat surface at (approx.) constant depth and
    wanting (X,Y) for *any* detected shape at that same Z_in.
    """
    if Z_in is None or Z_in <= 0:                   # Validate the provided depth
        return None                                 # Invalid depth → no result
    cx = (im_w / 2.0) if im_w is not None else CX   # Principal point x (prefer image center)
    cy = (im_h / 2.0) if im_h is not None else CY   # Principal point y (prefer image center)
    X = (u_px - cx) * Z_in / FX                     # Back-project pixel u to X at depth Z_in
    Y = (v_px - cy) * Z_in / FY                     # Back-project pixel v to Y at depth Z_in
    return (float(X), float(Y))                     # Return (X,Y) as floats
