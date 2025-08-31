# ==========================
# tracker.py (fully commented)
# ==========================
# A minimal centroid tracker that:
# - Assigns stable IDs across frames using nearest-neighbor association
# - Smooths center/radius with an exponential moving average (EMA)
# - Keeps short trails for visualization
# This is simple and fast, good enough for the challenge’s purposes.

# Import dataclass to define a container class for track state
from dataclasses import dataclass, field  # Provides automatic __init__ and nice defaults

# Import typing helpers for readability
from typing import Dict, List, Tuple, Optional  # Type hints for code clarity

@dataclass
class Track:
    """Represents a single tracked object/state."""
    track_id: int                                   # Unique ID for this track
    cx: float                                       # Smoothed x center (float)
    cy: float                                       # Smoothed y center (float)
    r_px: float                                     # Smoothed radius (float)
    lost: int = 0                                   # How many frames since last seen
    ema_alpha: float = 0.5                          # EMA factor, 0..1 (higher => follow new measurement more)
    trail: List[Tuple[int, int]] = field(default_factory=list)  # List of past (x,y) for trails
    score_ema: float = 0.0                          # Smoothed confidence score (optional)

    def update(self, cx: float, cy: float, r_px: float, keep_trail: int = 20, score: Optional[float] = None):
        """Update track with a new detection and apply EMA smoothing."""
        self.cx = self.ema_alpha * cx + (1.0 - self.ema_alpha) * self.cx  # EMA for x center
        self.cy = self.ema_alpha * cy + (1.0 - self.ema_alpha) * self.cy  # EMA for y center
        self.r_px = self.ema_alpha * r_px + (1.0 - self.ema_alpha) * self.r_px  # EMA for radius
        if score is not None:                                             # If a confidence score was provided
            self.score_ema = self.ema_alpha * score + (1.0 - self.ema_alpha) * self.score_ema  # EMA for score
        self.lost = 0                                                     # Reset lost counter on update
        self.trail.append((int(self.cx), int(self.cy)))                   # Add current point to trail
        if len(self.trail) > keep_trail:                                  # If trail too long
            self.trail.pop(0)                                             # Remove oldest point

class CentroidTracker:
    """Simple nearest-neighbor data association with distance gating."""
    def __init__(self, max_dist: float = 60.0, max_lost: int = 10, ema_alpha: float = 0.5, keep_trail: int = 20):
        self.max_dist = max_dist                 # Max pixel distance for associating detection to a track
        self.max_lost = max_lost                 # How many frames a track can be unmatched before drop
        self.ema_alpha = ema_alpha               # EMA parameter passed into new tracks
        self.keep_trail = keep_trail             # Number of points to keep for drawing trails
        self.next_id = 0                         # Next ID to assign for new track
        self.tracks: Dict[int, Track] = {}       # Dictionary mapping id -> Track

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance between two (x,y) points."""
        dx = a[0] - b[0]                         # Difference in x
        dy = a[1] - b[1]                         # Difference in y
        return (dx*dx + dy*dy) ** 0.5            # Square root of sum of squares

    def update(self, detections: List[dict]) -> Dict[int, Track]:
        """
        Update tracker with list of detections.
        Each detection dict is expected to have:
          - 'center_px': (x, y)
          - 'radius_px': float
          - 'score' (optional): float
        """
        det_centers = [(d['center_px'][0], d['center_px'][1], d['radius_px'], d.get('score', None)) for d in detections]  # Pack tuples
        for t in self.tracks.values():           # For each existing track
            t.lost += 1                          # Assume lost this frame until matched

        unmatched_dets = set(range(len(det_centers)))  # All detections start as unmatched
        used_tracks = set()                             # Track IDs used in this frame

        # Greedy nearest neighbor association
        for det_idx in list(unmatched_dets):           # Iterate over detection indices
            cx, cy, r, sc = det_centers[det_idx]       # Unpack a detection
            best_id = None                              # Best matching track id found so far
            best_dist = self.max_dist                   # Start with distance gate threshold
            for tid, tr in self.tracks.items():         # Try matching to each existing track
                if tr.track_id in used_tracks:          # Skip tracks already matched this frame
                    continue                            # Continue to next track
                dist = self._distance((cx, cy), (tr.cx, tr.cy))  # Compute distance to track
                if dist <= best_dist:                   # If within gate and closer than prior best
                    best_dist = dist                    # Update best distance
                    best_id = tid                       # Record best track id
            if best_id is not None:                     # If a match was found
                tr = self.tracks[best_id]               # Fetch the matching Track
                tr.update(cx, cy, r, keep_trail=self.keep_trail, score=sc)  # Update with detection
                used_tracks.add(best_id)                # Mark this track as used
                if det_idx in unmatched_dets:           # Remove from unmatched set
                    unmatched_dets.remove(det_idx)      # Now matched

        # Create new tracks for any detections that stayed unmatched
        for det_idx in unmatched_dets:                  # For each unmatched detection
            cx, cy, r, sc = det_centers[det_idx]        # Unpack detection data
            tr = Track(                                 # Create a new Track instance
                track_id=self.next_id,                  # Assign next available ID
                cx=cx, cy=cy, r_px=r,                  # Initialize with current detection
                ema_alpha=self.ema_alpha,               # Use configured EMA alpha
                score_ema=(sc or 0.0)                   # Initialize score EMA (0 if None)
            )
            tr.trail.append((int(cx), int(cy)))         # Start trail with first point
            self.tracks[self.next_id] = tr              # Register new track
            self.next_id += 1                           # Increment ID counter

        # Drop tracks that have been lost for too many frames
        drop_ids = [tid for tid, tr in self.tracks.items() if tr.lost > self.max_lost]  # Collect stale IDs
        for tid in drop_ids:                            # For each stale track ID
            del self.tracks[tid]                        # Remove from active tracks

        return self.tracks                              # Return dict of current tracks

