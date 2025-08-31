# ==========================
# detector.py (fully commented)
# ==========================
# Purpose:
#   Detect solid shapes, outline them, locate centers, compute a confidence
#   score, and assign a robust label:
#     "circle", "triangle", "quadrilateral", "pentagon", or "polygon".
#
# Robustness features:
#   - Multi-cue circle test (circularity, fill ratio, ellipse axis ratio, radial consistency)
#   - Polygon vertex counting on the CONVEX HULL with multi-epsilon approxPolyDP
#     and near-collinear vertex pruning
#   - Tie-breaking that prefers FEWER vertices on ambiguity (helps triangles not
#     collapse into quads in the hard video)
#   - Quad→Triangle rescue: if a 4-vertex polygon has a spurious tiny edge or a
#     nearly straight interior angle, we collapse it to a triangle
#   - Adaptive Sobel gradient threshold to reject textured grass (background agnostic)
#   - Gentle morphology (3×3) so we don’t blunt sharp corners
#
# NOTE (label flicker across frames):
#   This file processes ONE image at a time. For videos, you’ll still see some
#   label flicker as shapes move or blur. The right place to smooth labels
#   over time is your tracker (CentroidTracker) by keeping a running majority
#   vote of labels per track. Detector stays stateless by design.

import cv2                      # OpenCV: image ops, k-means, contours, drawing
import numpy as np              # NumPy: arrays & math
import math                     # math: pi, hypot, acos, etc.

# ----------------------------
# Helper: mean gradient inside a contour (lower => smoother/solid)
# ----------------------------
def _region_mean_grad(grad_mag, contour, shape):
    """
    Compute mean gradient magnitude within 'contour'.
    grad_mag : 2D float array of gradient magnitudes
    contour  : contour points (np.ndarray of shape Nx1x2)
    shape    : (H, W) reference to create a mask
    """
    mask = np.zeros(shape, np.uint8)                       # 1-channel mask (H×W), all zeros
    cv2.drawContours(mask, [contour], -1, 255, -1)         # Fill the contour region with 255
    vals = grad_mag[mask.astype(bool)]                     # Sample gradient magnitudes inside the region
    if vals.size == 0:                                     # Degenerate region guard
        return 1e9                                         # Return huge gradient => gets rejected later
    return float(vals.mean())                              # Return average interior gradient

# ----------------------------
# Helper: solidity = area / hull_area
# ----------------------------
def _solidity(contour):
    """Return solidity in [0,1]; 1 means perfectly solid (no dents/holes)."""
    area = cv2.contourArea(contour)                        # Raw contour area (px^2)
    if area <= 0:                                          # Degenerate contour
        return 0.0                                         # Treat as zero solidity
    hull = cv2.convexHull(contour)                         # Build convex hull contour
    hull_area = cv2.contourArea(hull)                      # Area of convex hull
    if hull_area <= 0:                                     # Degenerate hull guard
        return 0.0                                         # Treat as zero
    return float(area) / float(hull_area)                  # Ratio in [0,1]

# ----------------------------
# Helper: color variance inside region (higher => more textured)
# ----------------------------
def _region_color_variance(bgr, contour):
    """Average per-channel variance within the contour region (B,G,R)."""
    mask = np.zeros(bgr.shape[:2], np.uint8)               # Empty H×W mask
    cv2.drawContours(mask, [contour], -1, 255, -1)         # Fill region
    if mask.sum() == 0:                                    # Degenerate region
        return 1e9                                         # Large variance => reject
    region = bgr[mask.astype(bool)]                        # Pixels in region (N×3)
    var = region.astype(np.float32).var(axis=0).mean()     # Mean of per-channel variances
    return float(var)                                      # Scalar variance

# ----------------------------
# Helper: normalize area with a cap (avoid huge blobs dominating)
# ----------------------------
def _norm_area(area, frame_area, cap=0.02):
    """
    Normalize area by frame area; cap keeps one giant component from dominating.
    Returns a value in [0,1].
    """
    if frame_area <= 0:                                    # Guard against divide-by-zero
        return 0.0
    return min(area / frame_area, cap) / cap               # Scale into [0,1]

# ----------------------------
# Helper: confidence blended from area / solidity / inverse texture
# ----------------------------
def _score_detection(area, frame_area, solidity, mean_grad, grad_max_ref=80.0,
                     w_area=0.35, w_sol=0.35, w_tex=0.30):
    """
    Return confidence in [0,1] combining:
      - normalized area (bigger better)
      - solidity (higher better)
      - inverse texture (lower mean gradient better)
    """
    a = _norm_area(area, frame_area)                       # Area score in [0,1]
    s = max(0.0, min(1.0, float(solidity)))                # Clamp solidity into [0,1]
    t = 1.0 - max(0.0, min(1.0, mean_grad / grad_max_ref)) # Invert texture: low grad -> high score
    return float(w_area * a + w_sol * s + w_tex * t)       # Weighted sum

# ============================================================
# Robust polygon helpers (prevents triangle/pentagon -> quadrilateral)
# ============================================================

def _angle_degrees(a, b, c):
    """
    Return the internal angle at point b (degrees) for triangle (a, b, c).
    Large angles near 180 mean the middle vertex is nearly collinear.
    """
    ax, ay = float(a[0]), float(a[1])                      # Unpack A (float)
    bx, by = float(b[0]), float(b[1])                      # Unpack B (float)
    cx, cy = float(c[0]), float(c[1])                      # Unpack C (float)
    v1x, v1y = ax - bx, ay - by                            # Vector BA
    v2x, v2y = cx - bx, cy - by                            # Vector BC
    n1 = math.hypot(v1x, v1y) + 1e-9                       # Norm of BA (avoid zero)
    n2 = math.hypot(v2x, v2y) + 1e-9                       # Norm of BC (avoid zero)
    cosang = (v1x * v2x + v1y * v2y) / (n1 * n2)           # Cosine of angle at B
    cosang = max(-1.0, min(1.0, cosang))                   # Clamp numeric noise
    return math.degrees(math.acos(cosang))                 # Angle in degrees

def _remove_near_collinear(poly, angle_keep=170.0):
    """
    Remove vertices whose internal angle is >= angle_keep (≈ straight segments).
    'poly' is Nx1x2 (like approxPolyDP output). Returns possibly fewer points.
    """
    if len(poly) < 3:                                      # Not enough to prune
        return poly                                        # Return unchanged
    pts = [tuple(p[0]) for p in poly]                      # Convert to list of (x,y)
    keep = []                                              # Collect vertices to keep
    n = len(pts)                                           # Number of vertices
    for i in range(n):                                     # For each vertex
        a = pts[(i - 1) % n]                               # Previous (wrap-around)
        b = pts[i]                                         # Current
        c = pts[(i + 1) % n]                               # Next (wrap-around)
        ang = _angle_degrees(a, b, c)                      # Internal angle at b
        if ang < angle_keep:                               # Keep genuine corners (< ~170°)
            keep.append([b])                               # Store as [[x,y]]
    if len(keep) < 3:                                      # Ensure we still have a polygon
        return poly                                        # Too aggressive -> revert
    return np.array(keep, dtype=np.int32)                  # Return pruned polygon (Nx1x2)

# --- helper: side lengths of a polygon Nx1x2 (wrap-around) ---
def _edge_lengths(poly):
    """Return list of edge lengths (wrap-around) for polygon Nx1x2."""
    pts = [tuple(p[0]) for p in poly]                      # [(x,y), ...]
    n = len(pts)                                           # Vertex count
    if n < 2:                                              # Degenerate
        return []                                          # No edges
    lens = []                                              # Collected lengths
    for i in range(n):                                     # For each edge i->i+1
        x1, y1 = pts[i]                                    # Vertex i
        x2, y2 = pts[(i + 1) % n]                          # Vertex i+1 (wrap)
        lens.append(math.hypot(x2 - x1, y2 - y1))          # Euclidean length
    return lens                                            # Return lengths

# --- helper: internal angles (degrees) at each vertex (wrap-around) ---
def _internal_angles(poly):
    """Return list of internal angles (degrees) for polygon Nx1x2."""
    pts = [tuple(p[0]) for p in poly]                      # [(x,y), ...]
    n = len(pts)                                           # Vertex count
    if n < 3:                                              # Not a polygon
        return []                                          # No angles
    angs = []                                              # Collected angles
    for i in range(n):                                     # For each vertex
        a = pts[(i - 1) % n]                               # Prev
        b = pts[i]                                         # Curr
        c = pts[(i + 1) % n]                               # Next
        angs.append(_angle_degrees(a, b, c))               # Angle at b
    return angs                                            # Return angles

# --- if approx returns 4 points but it "looks like" a triangle, collapse it ---
def _maybe_collapse_quad_to_triangle(approx):
    """
    Heuristic: if a quadrilateral has one very short edge (spurious micro-corner)
    OR one interior angle is nearly straight, re-label as triangle.
    Returns True if it should be collapsed to a triangle.
    """
    if len(approx) != 4:                                   # Only applies to quads
        return False                                       # Not a quad -> ignore
    lens = _edge_lengths(approx)                           # Edge lengths
    angs = _internal_angles(approx)                        # Internal angles
    if not lens or not angs:                               # Degenerate guard
        return False                                       # Cannot decide

    shortest = min(lens)                                   # Smallest edge length
    median_len = sorted(lens)[len(lens)//2]                # Median edge length
    short_edge_ratio = (shortest / (median_len + 1e-9))    # Ratio of smallest-to-median

    max_angle = max(angs)                                  # Largest internal angle

    # Thresholds tuned for the "hard" video to rescue true triangles
    if short_edge_ratio < 0.22:                            # Very small edge present => merge
        return True
    if max_angle > 168.0:                                  # Nearly straight corner => merge
        return True
    return False                                           # Otherwise keep as quad

def _stable_poly_approx(hull_cnt,
                        eps_fracs=(0.003, 0.005, 0.008, 0.012, 0.016),
                        angle_keep=165.0):
    """
    Approximate polygon on the convex hull with multiple epsilon scales:
      - For each epsilon: approxPolyDP -> remove near-collinear points
      - Pick the candidate with the most common vertex count (mode).
        Tie-break **prefers smaller n** (helps triangles not become quads),
        then the smallest epsilon index.
    """
    perim = cv2.arcLength(hull_cnt, True)                  # Perimeter of convex hull
    cands = []                                             # Candidate polygons
    for ef in eps_fracs:                                   # Try several epsilons
        eps = max(1.0, perim * ef)                         # Avoid sub-pixel eps
        approx = cv2.approxPolyDP(hull_cnt, eps, True)     # Polygonal approximation
        approx = _remove_near_collinear(approx, angle_keep=angle_keep)  # Prune straight-ish corners
        cands.append(approx)                               # Save candidate

    counts = {}                                            # Vertex-count histogram
    for i, a in enumerate(cands):                          # Tally candidates
        n = len(a)                                         # Number of vertices
        counts.setdefault(n, []).append(i)                 # Record index for that n

    # Choose vertex count with biggest bucket; tie-break: prefer smaller n, then earliest epsilon
    best_n = min(counts.keys(), key=lambda k: (-len(counts[k]), k))  # more votes first, then fewer vertices
    best_idx = counts[best_n][0]                           # First epsilon with that n
    return cands[best_idx]                                 # Return best polygon

def _polygon_vertex_count(cnt):
    """
    Robust vertex count for a contour:
      1) convex hull
      2) multi-epsilon approx on the hull
      3) collinearity pruning (inside)
      4) quad→triangle rescue if needed
    Returns (n_vertices, approx_poly).
    """
    hull = cv2.convexHull(cnt)                             # Build convex hull
    approx = _stable_poly_approx(hull)                     # Stable polygon from hull
    if len(approx) == 4 and _maybe_collapse_quad_to_triangle(approx):  # Rescue triangles mislabeled as quads
        lens = _edge_lengths(approx)                       # Edge lengths
        i_short = int(np.argmin(lens))                     # Index of shortest edge
        keep = []                                          # Keep 3 vertices (drop one)
        for i in range(4):                                 # For each vertex
            if i == (i_short + 1) % 4:                     # Drop the latter endpoint of shortest edge
                continue
            keep.append([approx[i][0]])                    # Keep as [[x,y]]
        approx = np.array(keep, dtype=np.int32)            # Build new triangle
    return len(approx), approx                             # Return vertex count and polygon

# ============================================================
# Robust circle classifier (multi-cue + tolerant thresholds)
# ============================================================

def _is_circle_robust(cnt):
    """
    Decide if 'cnt' is a circle using multiple cues:
      - circularity (4πA / P²)
      - fill ratio (area vs enclosing circle)
      - ellipse axis ratio (minor/major)
      - radial consistency (std/mean distance to centroid)
    Slightly relaxed thresholds to reduce false negatives on hard footage.
    """
    area = cv2.contourArea(cnt)                            # Area of contour
    if area <= 0:                                          # Degenerate guard
        return False                                       # Not a circle

    peri = cv2.arcLength(cnt, True)                        # Perimeter
    circularity = (4.0 * math.pi * area) / (peri * peri + 1e-9)  # 1.0 for perfect circle

    (cx_c, cy_c), r_enclose = cv2.minEnclosingCircle(cnt)  # Enclosing circle (center, radius)
    fill_ratio = area / (math.pi * (r_enclose**2) + 1e-9)  # How tightly contour fills its enclosing circle

    axis_ratio = 1.0                                       # Default (perfectly round)
    if len(cnt) >= 5:                                      # fitEllipse requires ≥ 5 points
        ellipse = cv2.fitEllipse(cnt)                      # Fit ellipse to the contour
        (center_xy, (major, minor), _ang) = ellipse        # Extract major/minor axes
        if major > 1e-6:                                   # Avoid divide-by-zero
            axis_ratio = float(minor) / float(major)       # Ratio in [0,1]; 1 => circle

    M = cv2.moments(cnt)                                   # Moments for centroid
    if M["m00"] > 0:                                       # Valid area moment
        cx = M["m10"] / M["m00"]                           # Centroid x
        cy = M["m01"] / M["m00"]                           # Centroid y
    else:                                                  # Fallback to enclosing circle center
        (cx, cy) = (cx_c, cy_c)

    pts = cnt.reshape(-1, 2).astype(np.float32)            # Flatten to Nx2
    dists = []                                             # Distances to centroid
    for p in pts:                                          # For each contour point
        x, y = p[0], p[1]                                  # Unpack
        dists.append(math.hypot(x - cx, y - cy))           # Euclidean distance
    mean_r = (sum(dists) / (len(dists) + 1e-9)) if dists else 0.0  # Mean radius
    std_r  = (sum((di - mean_r)**2 for di in dists) / (len(dists) + 1e-9))**0.5 if dists else 0.0
    radial_cv = (std_r / (mean_r + 1e-9)) if mean_r > 0 else 1.0    # Coef. of variation

    # Relaxed-but-reasonable circle gates
    is_circle = (
        circularity >= 0.78 and                             # Round by perimeter metric
        fill_ratio  >= 0.80 and                             # Tightly fills enclosing circle
        axis_ratio  >= 0.78 and                             # Not elongated
        radial_cv   <= 0.18                                 # Radius fairly consistent
    )
    return is_circle                                       # True if passes all cues

# ----------------------------
# Label selection: circle first, then polygon count
# ----------------------------
def _label_shape(cnt):
    """
    Final labeling pipeline:
      1) Try robust circle test (so true circles don't become polygons).
      2) Else, robust vertex count on convex hull (with quad→triangle rescue).
    """
    if _is_circle_robust(cnt):                              # Circle guard first
        return "circle"                                     # Label circle
    n, _approx = _polygon_vertex_count(cnt)                 # Robust vertex count
    if n == 3:                                              # Map counts to shape names
        return "triangle"
    elif n == 4:
        return "quadrilateral"
    elif n == 5:
        return "pentagon"
    else:
        return "polygon"

# ----------------------------
# Main detection routine
# ----------------------------
def detect_shapes(
    bgr,                         # Input image (BGR)
    K=5,                         # k-means clusters in Lab color space
    min_area=300,                # Minimum area to keep (px^2)
    solidity_min=0.85,           # Minimum solidity
    use_adaptive_grad=True,      # Use gradient percentile as texture threshold
    grad_percentile=0.25,        # Percentile in [0,1] for threshold (e.g., 0.25 = 25th)
    grad_thresh_fallback=40.0,   # Fallback gradient threshold if not enough samples
    colorvar_max=None,           # Optional: reject high color variance regions
    use_ellipse_radius=False     # If True: use ellipse minor axis as radius (better for depth)
):
    """
    Detect shapes and return:
      annotated : BGR image with contours, centers, and labels
      detections: list of dicts with keys:
        - center_px: (x,y)
        - radius_px: float
        - contour  : np.ndarray of contour points
        - score    : confidence [0,1]
        - shape    : label string
    """
    H, W = bgr.shape[:2]                                          # Image height/width
    annotated = bgr.copy()                                        # Drawing buffer

    # --- gradient magnitude for texture gate (reject textured grass) ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)                  # BGR -> Gray
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)               # Sobel X
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)               # Sobel Y
    grad_mag = cv2.magnitude(gx, gy)                              # Gradient magnitude

    flat = grad_mag.reshape(-1)                                   # Flatten to 1D
    valid = flat[flat > 0]                                        # Ignore zeros
    if use_adaptive_grad and valid.size > 1000:                   # If enough samples for percentile
        grad_thresh = float(np.percentile(valid, grad_percentile * 100.0))  # Adaptive threshold
    else:                                                         # Otherwise use fallback constant
        grad_thresh = grad_thresh_fallback                        # Fixed threshold

    # --- color segmentation in Lab (background agnostic) ---
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)                    # BGR -> Lab
    Z = lab.reshape((-1, 3)).astype(np.float32)                   # Flatten to N×3 float32 for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)  # KMeans stop criteria
    _, labels, _ = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS) # Run k-means clustering
    labels = labels.reshape((H, W))                                # Reshape labels back to H×W

    detections = []                                                # Output detections
    frame_area = float(H * W)                                      # Total image area

    for k in range(K):                                             # For each cluster id
        mask = (labels == k).astype(np.uint8) * 255                # Binary mask for this cluster

        # Morphology (gentle): use 3×3 kernel to preserve triangle/pentagon corners
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 3×3 elliptical kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker, iterations=1)  # Remove tiny specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)  # Close tiny gaps

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # External contours
        for cnt in contours:                                       # Process each contour
            area = cv2.contourArea(cnt)                            # Area in px^2
            if area < min_area:                                    # Too small => ignore
                continue

            sol = _solidity(cnt)                                   # Solidity in [0,1]
            if sol < solidity_min:                                 # Not solid enough => skip
                continue

            mean_grad = _region_mean_grad(grad_mag, cnt, (H, W))   # Interior texture score
            if mean_grad > grad_thresh:                            # Too textured (likely background) => skip
                continue

            if colorvar_max is not None:                           # Optional color variance gate (if used)
                if _region_color_variance(bgr, cnt) > colorvar_max:
                    continue

            M = cv2.moments(cnt)                                   # Moments for centroid
            if M["m00"] != 0:                                      # Valid area moment?
                cx = int(M["m10"] / M["m00"])                      # Centroid x (int for drawing)
                cy = int(M["m01"] / M["m00"])                      # Centroid y
            else:                                                  # Fallback: enclosing circle center
                (cx_f, cy_f), _ = cv2.minEnclosingCircle(cnt)      # Center from enclosing circle
                cx, cy = int(cx_f), int(cy_f)                      # Convert to ints

            if use_ellipse_radius and len(cnt) >= 5:               # Depth-friendly radius (ellipse minor/2)
                ellipse = cv2.fitEllipse(cnt)                      # Fit ellipse
                (_cxy, (major, minor), _ang) = ellipse             # Axes lengths
                r_px = float(minor) / 2.0                          # Use minor axis / 2
            else:                                                  # Simpler: enclosing-circle radius
                (_c, _), r = cv2.minEnclosingCircle(cnt)           # Enclosing circle
                r_px = float(r)                                    # Radius in pixels

            shape_label = _label_shape(cnt)                        # Robust label (circle guard + hull vertex count)

            score = _score_detection(area, frame_area, sol, mean_grad)  # Confidence [0,1]

            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)      # Draw outline (green)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)         # Draw center dot (red)
            cv2.putText(annotated, f"({cx},{cy})", (cx + 8, cy - 8),    # Pixel coords
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, shape_label, (cx + 8, cy + 28),      # Shape label
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

            detections.append({                                        # Assemble detection record
                "center_px": (cx, cy),                                 # Pixel center (x,y)
                "radius_px": r_px,                                     # Pixel radius (float)
                "contour": cnt,                                        # Raw contour points
                "score": score,                                        # Confidence score
                "shape": shape_label,                                  # Label string
            })

    detections.sort(key=lambda d: d["center_px"][0])                   # Sort left→right for stable order
    return annotated, detections                                       # MUST return a 2-tuple
