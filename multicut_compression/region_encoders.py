"""
Region encoders to evaluate b(R) for MCCP.

We provide multiple encoding methods to satisfy the PDF requirement:
"Different methods for encoding the RGB values of image regions should be tested."

Encoders implemented here operate on a fixed segmentation (RegionRuns).

Encoders:
- MeanColor: lossy (stores 1 RGB per region)
- RLEPixels: lossless (RLE over bytes in region scan order)
- HuffmanPixels: lossless (Huffman over bytes in region scan order)
- PredictorZlibPixels: lossless (PNG-style predictor residuals per region + zlib)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import zlib
import heapq

from .fast_segmentation import RegionRuns


def iter_region_coords(region: RegionRuns) -> List[Tuple[int, int]]:
    y0, x0, _, _ = region.bbox
    coords: List[Tuple[int, int]] = []
    for dy, row_runs in region.runs:
        y = y0 + dy
        for rx, length in row_runs:
            x_start = x0 + rx
            for x in range(x_start, x_start + length):
                coords.append((y, x))
    return coords


def region_mask_in_bbox(region: RegionRuns) -> np.ndarray:
    y0, x0, h, w = region.bbox
    mask = np.zeros((h, w), dtype=bool)
    for dy, row_runs in region.runs:
        for rx, length in row_runs:
            mask[dy, rx:rx + length] = True
    return mask


def region_bytes(image: np.ndarray, region: RegionRuns) -> bytes:
    coords = iter_region_coords(region)
    if image.ndim == 2:
        arr = np.fromiter((int(image[y, x]) for y, x in coords), dtype=np.uint8)
        return arr.tobytes()
    # RGB
    arr = np.empty((len(coords) * 3,), dtype=np.uint8)
    k = 0
    for y, x in coords:
        px = image[y, x]
        arr[k:k+3] = px[:3]
        k += 3
    return arr.tobytes()


def bbox_runs_bits(region: RegionRuns) -> int:
    # bbox: 4*u16 = 64 bits; rows: u16 count + per-row (dy,u16 + num_runs,u16 + per-run 2*u16)
    bits = 64 + 16
    for _dy, row_runs in region.runs:
        bits += 32
        bits += len(row_runs) * 32
    return bits


class BaseRegionEncoder:
    name: str
    is_lossless: bool

    def region_bits(self, image: np.ndarray, region: RegionRuns) -> int:
        raise NotImplementedError


class MeanColorEncoder(BaseRegionEncoder):
    name = "MeanColor"
    is_lossless = False

    def region_bits(self, image: np.ndarray, region: RegionRuns) -> int:
        # shape + one RGB
        return bbox_runs_bits(region) + (24 if image.ndim == 3 else 8)


class RLEPixelsEncoder(BaseRegionEncoder):
    name = "RLE"
    is_lossless = True

    def region_bits(self, image: np.ndarray, region: RegionRuns) -> int:
        payload = region_bytes(image, region)
        if not payload:
            return bbox_runs_bits(region)
        # RLE encode bytes: (value,u8 + count,u16)
        out = bytearray()
        i = 0
        n = len(payload)
        while i < n:
            v = payload[i]
            j = i + 1
            while j < n and payload[j] == v and (j - i) < 65535:
                j += 1
            out.append(v)
            out.extend(int(j - i).to_bytes(2, "big"))
            i = j
        # Store encoded length + bytes (length,u32)
        return bbox_runs_bits(region) + 32 + len(out) * 8


class HuffmanPixelsEncoder(BaseRegionEncoder):
    name = "Huffman"
    is_lossless = True

    def region_bits(self, image: np.ndarray, region: RegionRuns) -> int:
        payload = region_bytes(image, region)
        if not payload:
            return bbox_runs_bits(region)

        # Frequency table (0..255)
        freq = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256).astype(np.int64)
        symbols = [(int(f), int(s)) for s, f in enumerate(freq) if f > 0]
        if len(symbols) == 1:
            # trivial: 1 bit per symbol
            encoded_bits = len(payload)  # 1 bit each (approx); store as bits, but will be byte-packed
            table_bits = 256 * 16  # store full freq table as u16 for simplicity
            return bbox_runs_bits(region) + table_bits + 32 + int((encoded_bits + 7) // 8) * 8

        # Use a stable tiebreaker so heap never compares int vs tuple nodes
        seq = 0
        heap = []
        for f, s in symbols:
            heap.append((f, seq, s))
            seq += 1
        heapq.heapify(heap)
        # Build tree
        while len(heap) > 1:
            f1, _s1, a = heapq.heappop(heap)
            f2, _s2, b = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, seq, (a, b)))
            seq += 1
        _f, _s, tree = heap[0]

        code_lens: Dict[int, int] = {}

        def walk(node, depth):
            if isinstance(node, int):
                code_lens[node] = max(1, depth)
                return
            left, right = node
            walk(left, depth + 1)
            walk(right, depth + 1)

        walk(tree, 0)

        # Total encoded bits = sum freq[s]*len(code[s])
        encoded_bits = int(sum(freq[s] * code_lens[s] for s in code_lens))

        # Store code lengths table as 256 u8 (simple, decodable)
        table_bits = 256 * 8
        return bbox_runs_bits(region) + table_bits + 32 + int((encoded_bits + 7) // 8) * 8


class PredictorZlibEncoder(BaseRegionEncoder):
    name = "Predictor+Zlib"
    is_lossless = True

    def region_bits(self, image: np.ndarray, region: RegionRuns) -> int:
        # Build best predictor residuals like in lossless_multicut, then zlib-compress residual bytes.
        y0, x0, rh, rw = region.bbox
        sub = image[y0:y0 + rh, x0:x0 + rw]
        channels = 1 if image.ndim == 2 else 3
        mask = region_mask_in_bbox(region)

        coords: List[Tuple[int, int]] = []
        for dy, row_runs in region.runs:
            for rx, length in row_runs:
                for x in range(rx, rx + length):
                    coords.append((dy, x))

        def get_val(yy, xx, cc):
            if yy < 0 or xx < 0 or yy >= rh or xx >= rw:
                return 0
            if not mask[yy, xx]:
                return 0
            if channels == 1:
                return int(sub[yy, xx])
            return int(sub[yy, xx, cc])

        def paeth(a, b, c):
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                return a
            if pb <= pc:
                return b
            return c

        best_cost = None
        best_bytes = b""
        best_pred = 0

        for pred_id in (0, 1, 2, 3):
            # int16 residuals
            res = np.empty((len(coords) * channels,), dtype=np.int16)
            cost = 0
            k = 0
            for yy, xx in coords:
                for cc in range(channels):
                    a = get_val(yy, xx - 1, cc)
                    b = get_val(yy - 1, xx, cc)
                    c = get_val(yy - 1, xx - 1, cc)
                    if pred_id == 0:
                        pred = a
                    elif pred_id == 1:
                        pred = b
                    elif pred_id == 2:
                        pred = (a + b) // 2
                    else:
                        pred = paeth(a, b, c)
                    orig = int(sub[yy, xx]) if channels == 1 else int(sub[yy, xx, cc])
                    r = orig - pred
                    res[k] = r
                    cost += abs(r)
                    k += 1

            raw = res.tobytes()
            comp = zlib.compress(raw, level=9)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_bytes = comp
                best_pred = pred_id

        # store predictor_id (8 bits) + len (32 bits) + payload
        return bbox_runs_bits(region) + 8 + 32 + len(best_bytes) * 8


def get_region_encoders() -> List[BaseRegionEncoder]:
    return [
        MeanColorEncoder(),
        RLEPixelsEncoder(),
        HuffmanPixelsEncoder(),
        PredictorZlibEncoder(),
    ]

