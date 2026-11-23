"""
Cost-aware greedy region merging.

Goal (per PDF): minimize total bit cost sum_R b(R).

We implement a practical greedy post-processing step:
merge adjacent regions if it reduces the encoded bit cost:

  merge(R1, R2) if b(R1 ∪ R2) < b(R1) + b(R2)

This is not an ILP multicut solver, but it makes segmentation explicitly cost-driven.
"""

from __future__ import annotations

from typing import Dict, Tuple, Callable, List, Set, Optional

import numpy as np

from .fast_segmentation import RegionRuns


def compute_adjacency(labels: np.ndarray) -> Set[Tuple[int, int]]:
    """Return set of unordered adjacent label pairs (4-neighborhood boundaries)."""
    pairs: Set[Tuple[int, int]] = set()

    # Right edges
    a = labels[:, :-1]
    b = labels[:, 1:]
    diff = a != b
    if np.any(diff):
        la = a[diff].ravel()
        lb = b[diff].ravel()
        for x, y in zip(la, lb):
            if x == y:
                continue
            x = int(x)
            y = int(y)
            pairs.add((x, y) if x < y else (y, x))

    # Down edges
    a = labels[:-1, :]
    b = labels[1:, :]
    diff = a != b
    if np.any(diff):
        la = a[diff].ravel()
        lb = b[diff].ravel()
        for x, y in zip(la, lb):
            if x == y:
                continue
            x = int(x)
            y = int(y)
            pairs.add((x, y) if x < y else (y, x))

    return pairs


def regions_from_labels(labels: np.ndarray) -> Dict[int, RegionRuns]:
    """Build RegionRuns for each label from a label map."""
    h, w = labels.shape
    regions: Dict[int, RegionRuns] = {}

    # Group x positions by (label, y)
    by_label_row: Dict[int, Dict[int, List[int]]] = {}
    for y in range(h):
        row = labels[y]
        uniq = np.unique(row)
        for lab in uniq:
            xs = np.where(row == lab)[0]
            if xs.size == 0:
                continue
            lab_i = int(lab)
            by_label_row.setdefault(lab_i, {})[y] = xs.tolist()

    for lab, rows_dict in by_label_row.items():
        ys = list(rows_dict.keys())
        y_min = min(ys)
        y_max = max(ys)
        x_min = min(min(xs) for xs in rows_dict.values())
        x_max = max(max(xs) for xs in rows_dict.values())
        bbox = (y_min, x_min, y_max - y_min + 1, x_max - x_min + 1)

        runs: List[Tuple[int, List[Tuple[int, int]]]] = []
        pixel_count = 0
        for y in range(y_min, y_max + 1):
            xs = rows_dict.get(y)
            if not xs:
                continue
            run_start = xs[0]
            run_len = 1
            row_runs: List[Tuple[int, int]] = []
            for i in range(1, len(xs)):
                if xs[i] == xs[i - 1] + 1:
                    run_len += 1
                else:
                    row_runs.append((run_start - x_min, run_len))
                    pixel_count += run_len
                    run_start = xs[i]
                    run_len = 1
            row_runs.append((run_start - x_min, run_len))
            pixel_count += run_len
            runs.append((y - y_min, row_runs))

        regions[lab] = RegionRuns(bbox=bbox, runs=runs, pixel_count=int(pixel_count))

    return regions


def greedy_cost_aware_merge(
    image: np.ndarray,
    labels: np.ndarray,
    bit_cost_fn: Callable[[np.ndarray, RegionRuns], int],
    max_merges: int = 300,
    min_saving_bits: int = 32,
    max_pairs_per_iter: int = 300,
) -> Tuple[np.ndarray, Dict[int, RegionRuns], Dict[str, int]]:
    """
    Greedily merge adjacent regions to reduce total bit cost.

    bit_cost_fn should return bits required for encoding the region (excluding global headers).
    """
    labels = labels.copy()
    regions = regions_from_labels(labels)
    merges_done = 0
    total_saving = 0

    for _ in range(max_merges):
        pairs_set = compute_adjacency(labels)
        if not pairs_set:
            break

        # Evaluate only a capped number of candidate neighbor pairs (keeps runtime bounded)
        pairs = list(pairs_set)
        if len(pairs) > max_pairs_per_iter:
            pairs = pairs[:max_pairs_per_iter]

        best_pair = None  # (a, b, merged_region)
        best_saving = 0

        for a, b in pairs:
            ra = regions.get(a)
            rb = regions.get(b)
            if ra is None or rb is None:
                continue

            cost_a = int(bit_cost_fn(image, ra))
            cost_b = int(bit_cost_fn(image, rb))

            # Bounding box union
            y0 = min(ra.bbox[0], rb.bbox[0])
            x0 = min(ra.bbox[1], rb.bbox[1])
            y1 = max(ra.bbox[0] + ra.bbox[2], rb.bbox[0] + rb.bbox[2])
            x1 = max(ra.bbox[1] + ra.bbox[3], rb.bbox[1] + rb.bbox[3])

            sub = labels[y0:y1, x0:x1]
            merged_mask = (sub == a) | (sub == b)
            if not np.any(merged_mask):
                continue

            runs: List[Tuple[int, List[Tuple[int, int]]]] = []
            pixel_count = 0
            for dy in range(merged_mask.shape[0]):
                xs = np.where(merged_mask[dy])[0]
                if xs.size == 0:
                    continue
                xs = xs.tolist()
                row_runs: List[Tuple[int, int]] = []
                run_start = xs[0]
                run_len = 1
                for i in range(1, len(xs)):
                    if xs[i] == xs[i - 1] + 1:
                        run_len += 1
                    else:
                        row_runs.append((run_start, run_len))
                        pixel_count += run_len
                        run_start = xs[i]
                        run_len = 1
                row_runs.append((run_start, run_len))
                pixel_count += run_len
                runs.append((int(dy), [(int(rx), int(ln)) for rx, ln in row_runs]))

            merged_region = RegionRuns(
                bbox=(int(y0), int(x0), int(y1 - y0), int(x1 - x0)),
                runs=runs,
                pixel_count=int(pixel_count),
            )

            cost_m = int(bit_cost_fn(image, merged_region))
            saving = (cost_a + cost_b) - cost_m
            if saving > best_saving:
                best_saving = int(saving)
                best_pair = (a, b)

        if best_pair is None or best_saving < min_saving_bits:
            break

        a, b = best_pair
        labels[labels == b] = a
        regions = regions_from_labels(labels)
        merges_done += 1
        total_saving += best_saving

    stats = {"merges_done": int(merges_done), "total_saving_bits": int(total_saving)}
    return labels, regions, stats


def _region_overhead_bits(region: RegionRuns) -> int:
    # bbox: 4*u16 = 64 bits; rows count u16; per-row dy+num_runs (32 bits) + per-run (32 bits)
    bits = 64 + 16
    for _dy, row_runs in region.runs:
        bits += 32
        bits += len(row_runs) * 32
    return bits


def _compute_mean_color(image: np.ndarray, region: RegionRuns) -> np.ndarray:
    """Compute mean color (RGB) or grayscale mean as length-1 array."""
    y0, x0, _, _ = region.bbox
    if image.ndim == 2:
        s = 0
        n = 0
        for dy, row_runs in region.runs:
            y = y0 + dy
            for rx, length in row_runs:
                xs = x0 + rx
                xe = xs + length
                s += int(np.sum(image[y, xs:xe]))
                n += length
        return np.array([s / max(1, n)], dtype=np.float32)

    s = np.zeros((3,), dtype=np.float64)
    n = 0
    for dy, row_runs in region.runs:
        y = y0 + dy
        for rx, length in row_runs:
            xs = x0 + rx
            xe = xs + length
            strip = image[y, xs:xe, :3]
            s += np.sum(strip, axis=0)
            n += length
    if n <= 0:
        return np.zeros((3,), dtype=np.float32)
    return (s / n).astype(np.float32)


def _boundary_pairs_with_counts(labels: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Return dict of adjacent label pairs -> boundary edge count."""
    counts: Dict[Tuple[int, int], int] = {}

    # right boundaries
    a = labels[:, :-1]
    b = labels[:, 1:]
    diff = a != b
    if np.any(diff):
        la = a[diff].ravel()
        lb = b[diff].ravel()
        for x, y in zip(la, lb):
            x = int(x)
            y = int(y)
            if x == y:
                continue
            p = (x, y) if x < y else (y, x)
            counts[p] = counts.get(p, 0) + 1

    # down boundaries
    a = labels[:-1, :]
    b = labels[1:, :]
    diff = a != b
    if np.any(diff):
        la = a[diff].ravel()
        lb = b[diff].ravel()
        for x, y in zip(la, lb):
            x = int(x)
            y = int(y)
            if x == y:
                continue
            p = (x, y) if x < y else (y, x)
            counts[p] = counts.get(p, 0) + 1

    return counts


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def single_pass_cost_aware_merge(
    image: np.ndarray,
    labels: np.ndarray,
    regions: Dict[int, RegionRuns],
    max_merges: int = 5000,
    gamma_bits_per_unit: float = 24.0,
) -> Tuple[np.ndarray, Dict[int, RegionRuns], Dict[str, int]]:
    """
    Fast cost-aware merging (single pass, union-find).

    We compute an approximate bit-cost objective:
      b(R) ≈ overhead(shape)   and merging penalty ≈ boundary_len * color_distance * gamma

    Merge if:
      overhead(Ra)+overhead(Rb) - overhead(Runion) > penalty

    This is efficient and still explicitly b(R)-driven.
    """
    labels = labels.copy()
    max_label = int(labels.max()) if labels.size else -1
    n_regions = max_label + 1
    dsu = _DSU(n_regions)

    # Precompute per-region stats
    overhead = [0] * n_regions
    pixel_count = [0] * n_regions
    mean = [None] * n_regions  # np.ndarray
    bbox = [None] * n_regions  # (y0,x0,h,w)
    run_counts = [0] * n_regions
    row_counts = [0] * n_regions

    for rid, reg in regions.items():
        overhead[rid] = _region_overhead_bits(reg)
        pixel_count[rid] = int(reg.pixel_count)
        mean[rid] = _compute_mean_color(image, reg)
        bbox[rid] = reg.bbox
        row_counts[rid] = len(reg.runs)
        run_counts[rid] = int(sum(len(rr) for _dy, rr in reg.runs))

    # Candidate pairs with boundary counts
    pair_counts = _boundary_pairs_with_counts(labels)
    # Sort candidates by boundary length (most impactful merges first)
    candidates = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)

    merges_done = 0
    total_saving = 0

    def union_overhead_bits(ra: int, rb: int) -> int:
        # Approx union overhead using bbox union + (rows/runs sum) as upper bound.
        ya, xa, ha, wa = bbox[ra]
        yb, xb, hb, wb = bbox[rb]
        y0 = min(ya, yb)
        x0 = min(xa, xb)
        y1 = max(ya + ha, yb + hb)
        x1 = max(xa + wa, xb + wb)
        # bbox bits constant + approximate rows/runs metadata
        bits = 64 + 16
        bits += (row_counts[ra] + row_counts[rb]) * 32
        bits += (run_counts[ra] + run_counts[rb]) * 32
        # BBox size doesn’t directly change bits, but is captured by constant bbox bits.
        _ = (y0, x0, y1 - y0, x1 - x0)
        return int(bits)

    for (a, b), boundary_len in candidates:
        if merges_done >= max_merges:
            break
        ra = dsu.find(a)
        rb = dsu.find(b)
        if ra == rb:
            continue

        # Similarity penalty via mean color distance
        ma = mean[ra]
        mb = mean[rb]
        if ma is None or mb is None:
            continue
        if ma.size == 1:
            dist = abs(float(ma[0]) - float(mb[0])) / 255.0
        else:
            dist = float(np.linalg.norm(ma - mb)) / (255.0 * np.sqrt(3))

        penalty = int(boundary_len * dist * gamma_bits_per_unit)
        cost_union = union_overhead_bits(ra, rb)
        saving = (overhead[ra] + overhead[rb]) - cost_union - penalty
        if saving <= 0:
            continue

        root = dsu.union(ra, rb)
        other = rb if root == ra else ra

        # Update stats on root (approx)
        overhead[root] = cost_union
        pixel_count[root] += pixel_count[other]
        # weighted mean
        wa = pixel_count[root]
        wb = pixel_count[other]
        mean[root] = (mean[root] * wa + mean[other] * wb) / max(1.0, float(wa + wb))
        # bbox union
        ya, xa, ha, wa_ = bbox[root]
        yb, xb, hb, wb_ = bbox[other]
        y0 = min(ya, yb)
        x0 = min(xa, xb)
        y1 = max(ya + ha, yb + hb)
        x1 = max(xa + wa_, xb + wb_)
        bbox[root] = (y0, x0, y1 - y0, x1 - x0)
        row_counts[root] += row_counts[other]
        run_counts[root] += run_counts[other]

        merges_done += 1
        total_saving += int(saving)

    # Apply final label mapping
    root_map = np.vectorize(lambda x: dsu.find(int(x)))(labels).astype(np.int32)
    # Normalize to contiguous ids
    unique = np.unique(root_map)
    remap = {int(old): i for i, old in enumerate(unique)}
    for old, new in remap.items():
        if old != new:
            root_map[root_map == old] = new

    new_regions = regions_from_labels(root_map)
    stats = {"merges_done": int(merges_done), "total_saving_bits": int(total_saving)}
    return root_map, new_regions, stats

