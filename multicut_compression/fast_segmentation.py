"""
Fast region segmentation (no NetworkX).

This replaces the old NetworkX pixel-graph + region-growing approach with a
pure NumPy/queue-based flood-fill on the pixel grid.

Key properties:
- 4-connected region growing
- Optional adaptive thresholding using local gradient magnitude
- Deterministic (scanline seed order)

Output:
- labels: (H, W) int32 region-id per pixel
- regions: dict[region_id] -> region metadata with bbox + row-runs


"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


@dataclass(frozen=True)
class RegionRuns:
    """
    A compact region representation using a bounding box and row runs.

    bbox is (y0, x0, h, w).
    runs is a list of rows, each row is:
      (dy, [(x_start, length), ...]) where dy is relative to y0 and x_start is relative to x0.
    """

    bbox: Tuple[int, int, int, int]
    runs: List[Tuple[int, List[Tuple[int, int]]]]
    pixel_count: int


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    # Fallback: normalize by max if known-ish, else just float
    img = image.astype(np.float32)
    m = float(np.max(img)) if img.size else 1.0
    return img / m if m > 0 else img


def _similarity_rgb01(a: np.ndarray, b: np.ndarray) -> float:
    # Euclidean distance in [0, sqrt(C)] -> similarity in [0,1]
    diff = a - b
    dist = float(np.sqrt(np.sum(diff * diff)))
    max_dist = float(np.sqrt(a.shape[0]))
    return max(0.0, 1.0 - dist / max_dist) if max_dist > 0 else 1.0


def _compute_gradient_strength(gray01: np.ndarray) -> np.ndarray:
    # Sobel gradient magnitude in [0, ~] then normalize to [0,1]
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_max = float(np.max(mag)) if mag.size else 0.0
    if mag_max <= 0:
        return np.zeros_like(gray01, dtype=np.float32)
    return (mag / mag_max).astype(np.float32)


class FastRegionSegmenter:
    """
    Queue-based region growing on the pixel grid.

    similarity_threshold is in [0,1]. Higher means stricter similarity (more regions).

    If adaptive=True, the effective threshold becomes higher near edges/textures and lower
    in flat areas, using a gradient-derived map.
    """

    def __init__(self):
        pass

    def segment(
        self,
        image: np.ndarray,
        similarity_threshold: float = 0.85,
        min_region_size: int = 1,
        adaptive: bool = True,
        adaptive_strength: float = 0.20,
        min_threshold: float = 0.70,
        max_threshold: float = 0.98,
    ) -> Tuple[np.ndarray, Dict[int, RegionRuns]]:
        """
        Segment an image into regions.

        Returns:
            labels: (H, W) int32
            regions: dict region_id -> RegionRuns
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape: {image.shape}")

        img01 = _to_float01(image)
        h, w = img01.shape[:2]
        c = 1 if img01.ndim == 2 else img01.shape[2]

        labels = np.full((h, w), -1, dtype=np.int32)

        # Adaptive threshold map
        thr_map: Optional[np.ndarray] = None
        if adaptive:
            if img01.ndim == 3:
                gray = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                gray = img01.astype(np.float32)
            grad = _compute_gradient_strength(gray)
            # Low gradient -> lower effective threshold (merge more). High gradient -> higher threshold.
            thr_map = np.clip(
                similarity_threshold + adaptive_strength * grad,
                min_threshold,
                max_threshold,
            ).astype(np.float32)

        region_id = 0
        regions: Dict[int, RegionRuns] = {}

        # BFS helper
        q = deque()

        for y0 in range(h):
            for x0 in range(w):
                if labels[y0, x0] != -1:
                    continue

                # Start new region
                labels[y0, x0] = region_id
                q.clear()
                q.append((y0, x0))

                # Running mean for region (in float01)
                if c == 1:
                    mean = float(img01[y0, x0])
                else:
                    mean = img01[y0, x0].copy()
                count = 1

                # Track coords to build runs
                coords_by_row: Dict[int, List[int]] = defaultdict(list)
                coords_by_row[y0].append(x0)

                # Grow region
                while q:
                    y, x = q.popleft()

                    # 4-neighbors
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if labels[ny, nx] != -1:
                            continue

                        # Similarity to current region mean
                        if c == 1:
                            sim = 1.0 - abs(float(img01[ny, nx]) - float(mean))
                        else:
                            sim = _similarity_rgb01(img01[ny, nx], mean if isinstance(mean, np.ndarray) else img01[y0, x0])

                        thr = similarity_threshold
                        if thr_map is not None:
                            thr = float(thr_map[ny, nx])

                        if sim >= thr:
                            labels[ny, nx] = region_id
                            q.append((ny, nx))
                            coords_by_row[ny].append(nx)

                            # Update running mean
                            if c == 1:
                                mean = (mean * count + float(img01[ny, nx])) / (count + 1)
                            else:
                                mean = (mean * count + img01[ny, nx]) / (count + 1)
                            count += 1

                # Enforce min_region_size by leaving as-is (we don’t merge small regions here for determinism).
                # Callers can optionally post-process if needed.
                if count < max(1, min_region_size):
                    # Still store it, but caller may choose to merge or keep.
                    pass

                # Build compact runs
                ys = list(coords_by_row.keys())
                y_min = min(ys)
                y_max = max(ys)
                x_min = min(min(xs) for xs in coords_by_row.values())
                x_max = max(max(xs) for xs in coords_by_row.values())
                bbox = (y_min, x_min, y_max - y_min + 1, x_max - x_min + 1)

                runs: List[Tuple[int, List[Tuple[int, int]]]] = []
                pixel_count = 0

                for y in range(y_min, y_max + 1):
                    xs = coords_by_row.get(y)
                    if not xs:
                        continue
                    xs_sorted = sorted(xs)
                    row_runs: List[Tuple[int, int]] = []
                    run_start = xs_sorted[0]
                    run_len = 1
                    for i in range(1, len(xs_sorted)):
                        if xs_sorted[i] == xs_sorted[i - 1] + 1:
                            run_len += 1
                        else:
                            row_runs.append((run_start - x_min, run_len))
                            pixel_count += run_len
                            run_start = xs_sorted[i]
                            run_len = 1
                    row_runs.append((run_start - x_min, run_len))
                    pixel_count += run_len
                    runs.append((y - y_min, row_runs))

                regions[region_id] = RegionRuns(bbox=bbox, runs=runs, pixel_count=pixel_count)
                region_id += 1

        # Normalize labels to contiguous 0..K-1 for stable downstream processing
        unique = np.unique(labels)
        remap = {int(old): i for i, old in enumerate(unique)}
        for old, new in remap.items():
            if old != new:
                labels[labels == old] = new
        regions = {remap[int(k)]: v for k, v in regions.items()}

        return labels, regions

