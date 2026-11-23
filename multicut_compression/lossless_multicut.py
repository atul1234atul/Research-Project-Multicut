"""
Lossless Multicut Compression

This module implements TRUE LOSSLESS compression using:
1. Graph-based region segmentation (multicut)
2. Efficient encoding of ALL pixel values within each region

The key insight: Pixels within a region are similar, so delta encoding
and compression is efficient. We store ALL pixels (lossless).

"""

import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Any, Optional
import pickle
import zlib
import struct
from dataclasses import dataclass
import warnings

from .multicut_compressor import calculate_original_size_bits
from .fast_segmentation import FastRegionSegmenter, RegionRuns
from .cost_aware_merge import single_pass_cost_aware_merge
from .region_encoders import PredictorZlibEncoder, bbox_runs_bits

warnings.filterwarnings('ignore')


@dataclass
class LosslessCompressionResult:
    """Result of lossless multicut compression"""
    original_size: int          # Original size in bits
    compressed_size: int        # Compressed size in bits
    compression_ratio: float    # original_size / compressed_size
    computation_time: float     # Time in seconds
    num_regions: int            # Number of regions found
    image_shape: Tuple[int, ...]  # Original image shape
    encoded_data: bytes         # All encoded data
    is_lossless: bool = True    # Always True for this method
    encoding_method: str = 'Zlib+Delta'


class LosslessMulticutCompressor:
    """
    Lossless image compression using multicut segmentation.
    
    This method:
    1. Segments image into regions based on pixel similarity
    2. Encodes ALL pixel values within each region using delta + zlib
    3. Achieves lossless compression by exploiting spatial redundancy
    
    NOTE: Lower similarity_threshold = fewer regions = better compression.
    Threshold 0.70-0.80 is recommended for best compression.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        min_region_size: int = 1,
        adaptive_segmentation: bool = True,
        cost_aware: bool = True,
        # Keep this small so full datasets remain runnable.
        # This still satisfies the PDF requirement (b(R)-driven merges), but stays bounded.
        max_cost_merges: int = 10,
    ):
        """
        Args:
            similarity_threshold: Threshold for merging pixels (0-1)
            min_region_size: Minimum pixels per region
        """
        self.segmenter = FastRegionSegmenter()
        self.similarity_threshold = similarity_threshold
        self.min_region_size = min_region_size
        self.adaptive_segmentation = adaptive_segmentation
        self.cost_aware = cost_aware
        self.max_cost_merges = max_cost_merges
        
        self.image_shape = None
        self.regions = None
    
    def compress(self, image: np.ndarray) -> LosslessCompressionResult:
        """
        Compress image losslessly using multicut + efficient encoding.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            LosslessCompressionResult with compression metrics
        """
        start_time = time.time()
        self.image_shape = image.shape
        
        # Calculate correct original size
        original_size = calculate_original_size_bits(image)
        
        # Step 1: Segment into regions (fast, no NetworkX)
        _labels, regions = self.segmenter.segment(
            image,
            similarity_threshold=self.similarity_threshold,
            min_region_size=self.min_region_size,
            adaptive=self.adaptive_segmentation,
        )
        # Optional cost-aware merge to minimize sum b(R)
        if self.cost_aware:
            # Fast, explicit b(R)-driven merge pass (bounded + scalable)
            _labels, regions, _merge_stats = single_pass_cost_aware_merge(
                image=image,
                labels=_labels,
                regions=regions,
                max_merges=5000,
                gamma_bits_per_unit=24.0,
            )

        self.regions = regions
        
        # Step 3: Encode losslessly (bbox+runs + predictor residuals + zlib)
        encoded_data = self._encode_lossless(image, regions)
        
        # Calculate compressed size
        compressed_size = len(encoded_data) * 8
        
        computation_time = time.time() - start_time
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return LosslessCompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            computation_time=computation_time,
            num_regions=len(regions),
            image_shape=image.shape,
            encoded_data=encoded_data,
            is_lossless=True,
            encoding_method='Predictor+Zlib'
        )
    
    def _encode_lossless(self, image: np.ndarray, regions: Dict[int, RegionRuns]) -> bytes:
        """
        Encode all pixels losslessly using:
        - compact region shape: bbox + row runs (spatially aware)
        - predictor-based residuals (PNG-style) within each region
        - zlib compression over the whole stream
        
        Strategy:
        1. Group pixels by region
        2. For each region, store pixel positions and values
        3. Use delta encoding within regions (similar pixels = small deltas)
        4. Compress everything with zlib
        """
        data = bytearray()
        
        # Header: image shape
        if len(image.shape) == 3:
            height, width, channels = image.shape
            data.extend(struct.pack('>III', height, width, channels))
        else:
            height, width = image.shape
            channels = 1
            data.extend(struct.pack('>II', height, width))
        
        data.extend(struct.pack('>B', channels))
        
        # Number of regions
        data.extend(struct.pack('>I', len(regions)))
        
        # Encode each region
        for region_id, region in regions.items():
            region_data = bytearray()
            
            # Region ID
            region_data.extend(struct.pack('>I', region_id))

            # bbox + runs
            y0, x0, rh, rw = region.bbox
            if y0 > 65535 or x0 > 65535 or rh > 65535 or rw > 65535:
                raise ValueError("Image too large for u16 bbox encoding")
            region_data.extend(struct.pack('>HHHH', y0, x0, rh, rw))

            rows = region.runs
            region_data.extend(struct.pack('>H', len(rows)))
            for dy, row_runs in rows:
                region_data.extend(struct.pack('>HH', dy, len(row_runs)))
                for rx, length in row_runs:
                    region_data.extend(struct.pack('>HH', rx, length))

            # Predictor residuals
            predictor_id, residuals_i16 = self._encode_region_residuals(image, region, channels)
            region_data.extend(struct.pack('>B', predictor_id))
            region_data.extend(struct.pack('>I', residuals_i16.size))  # number of int16 values
            region_data.extend(residuals_i16.tobytes(order='C'))
            
            data.extend(region_data)
        
        # Compress everything with zlib
        compressed = zlib.compress(bytes(data), level=9)
        
        return compressed
    
    def _encode_region_residuals(self, image: np.ndarray, region: RegionRuns, channels: int) -> Tuple[int, np.ndarray]:
        """
        Compute predictor-based residuals for a region and pick the best predictor.

        Predictors (PNG-style):
          0: Left
          1: Up
          2: Average
          3: Paeth

        Residuals are stored as int16 (orig - pred), lossless.
        """
        y0, x0, rh, rw = region.bbox
        sub = image[y0:y0 + rh, x0:x0 + rw]

        # Build mask for membership within bbox
        mask = np.zeros((rh, rw), dtype=bool)
        coords: List[Tuple[int, int]] = []
        for dy, row_runs in region.runs:
            for rx, length in row_runs:
                mask[dy, rx:rx + length] = True
                for x in range(rx, rx + length):
                    coords.append((dy, x))

        # Helper to get neighbor value (only if neighbor is in region)
        def get_val(arr, yy, xx, cc):
            if yy < 0 or xx < 0 or yy >= rh or xx >= rw:
                return 0
            if not mask[yy, xx]:
                return 0
            if channels == 1:
                return int(arr[yy, xx])
            return int(arr[yy, xx, cc])

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

        best_pred = 0
        best_cost = None
        best_residuals: Optional[np.ndarray] = None

        for pred_id in (0, 1, 2, 3):
            res = np.empty((len(coords) * channels,), dtype=np.int16)
            cost = 0
            k = 0
            for yy, xx in coords:
                for cc in range(channels):
                    a = get_val(sub, yy, xx - 1, cc)
                    b = get_val(sub, yy - 1, xx, cc)
                    c = get_val(sub, yy - 1, xx - 1, cc)
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

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pred = pred_id
                best_residuals = res

        assert best_residuals is not None
        return best_pred, best_residuals
    
    def decompress(self, result: LosslessCompressionResult) -> np.ndarray:
        """Decompress image from compression result (lossless)"""
        # Decompress zlib
        data = zlib.decompress(result.encoded_data)
        offset = 0
        
        # Read header
        if len(result.image_shape) == 3:
            height, width, channels = struct.unpack('>III', data[offset:offset+12])
            offset += 12
        else:
            height, width = struct.unpack('>II', data[offset:offset+8])
            channels = 1
            offset += 8
        
        channels_check = struct.unpack('>B', data[offset:offset+1])[0]
        offset += 1
        
        num_regions = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        # Create output image
        if channels > 1:
            reconstructed = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            reconstructed = np.zeros((height, width), dtype=np.uint8)
        
        # Decode each region
        for _ in range(num_regions):
            _region_id = struct.unpack('>I', data[offset:offset + 4])[0]
            offset += 4

            y0, x0, rh, rw = struct.unpack('>HHHH', data[offset:offset + 8])
            offset += 8

            # Runs
            num_rows = struct.unpack('>H', data[offset:offset + 2])[0]
            offset += 2
            runs: List[Tuple[int, List[Tuple[int, int]]]] = []
            pixel_count = 0
            for _ in range(num_rows):
                dy, num_runs = struct.unpack('>HH', data[offset:offset + 4])
                offset += 4
                row_runs: List[Tuple[int, int]] = []
                for _ in range(num_runs):
                    rx, length = struct.unpack('>HH', data[offset:offset + 4])
                    offset += 4
                    row_runs.append((rx, length))
                    pixel_count += length
                runs.append((dy, row_runs))

            predictor_id = struct.unpack('>B', data[offset:offset + 1])[0]
            offset += 1

            n_i16 = struct.unpack('>I', data[offset:offset + 4])[0]
            offset += 4
            byte_len = n_i16 * 2
            residuals = np.frombuffer(data[offset:offset + byte_len], dtype=np.int16).copy()
            offset += byte_len

            # Reconstruct region pixels from residuals using same predictor
            self._decode_region_into(
                reconstructed=reconstructed,
                y0=y0,
                x0=x0,
                rh=rh,
                rw=rw,
                runs=runs,
                residuals=residuals,
                predictor_id=predictor_id,
                channels=channels,
            )
        
        return reconstructed

    def _decode_region_into(
        self,
        reconstructed: np.ndarray,
        y0: int,
        x0: int,
        rh: int,
        rw: int,
        runs: List[Tuple[int, List[Tuple[int, int]]]],
        residuals: np.ndarray,
        predictor_id: int,
        channels: int,
    ) -> None:
        # Build mask
        mask = np.zeros((rh, rw), dtype=bool)
        coords: List[Tuple[int, int]] = []
        for dy, row_runs in runs:
            for rx, length in row_runs:
                mask[dy, rx:rx + length] = True
                for x in range(rx, rx + length):
                    coords.append((dy, x))

        def get_val(yy, xx, cc):
            if yy < 0 or xx < 0 or yy >= rh or xx >= rw:
                return 0
            if not mask[yy, xx]:
                return 0
            if channels == 1:
                return int(reconstructed[y0 + yy, x0 + xx])
            return int(reconstructed[y0 + yy, x0 + xx, cc])

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

        k = 0
        for yy, xx in coords:
            if channels == 1:
                a = get_val(yy, xx - 1, 0)
                b = get_val(yy - 1, xx, 0)
                c = get_val(yy - 1, xx - 1, 0)
                if predictor_id == 0:
                    pred = a
                elif predictor_id == 1:
                    pred = b
                elif predictor_id == 2:
                    pred = (a + b) // 2
                else:
                    pred = paeth(a, b, c)
                val = int(pred + int(residuals[k]))
                reconstructed[y0 + yy, x0 + xx] = np.uint8(val & 0xFF)
                k += 1
            else:
                for cc in range(channels):
                    a = get_val(yy, xx - 1, cc)
                    b = get_val(yy - 1, xx, cc)
                    c = get_val(yy - 1, xx - 1, cc)
                    if predictor_id == 0:
                        pred = a
                    elif predictor_id == 1:
                        pred = b
                    elif predictor_id == 2:
                        pred = (a + b) // 2
                    else:
                        pred = paeth(a, b, c)
                    val = int(pred + int(residuals[k]))
                    reconstructed[y0 + yy, x0 + xx, cc] = np.uint8(val & 0xFF)
                    k += 1
    
    def save_compressed(self, result: LosslessCompressionResult, filename: str):
        """Save compressed data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
    
    def load_compressed(self, filename: str) -> LosslessCompressionResult:
        """Load compressed data from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def verify_lossless(original: np.ndarray, reconstructed: np.ndarray) -> bool:
    """Verify that compression is truly lossless"""
    return np.array_equal(original, reconstructed)
