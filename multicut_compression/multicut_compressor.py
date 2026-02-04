"""
Region-Based Lossy Image Compression via Graph Partitioning

This module implements a lossy image compression pipeline using graph-based
region segmentation. Each region is represented by a single representative
color value, achieving compression through spatial redundancy.

IMPORTANT: This is a LOSSY compression method. It does NOT preserve exact
pixel values. Use PSNR/SSIM metrics to evaluate quality.

Based on graph partitioning concepts, but uses a simplified region-growing
approach rather than true multicut optimization.


"""

import numpy as np
import cv2
from collections import Counter
import time
from typing import Dict, List, Tuple, Any, Optional
import pickle
import zlib
import struct
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from .fast_segmentation import FastRegionSegmenter, RegionRuns


def calculate_original_size_bits(image: np.ndarray) -> int:
    """
    Calculate the correct original size in bits.
    
    For an image with shape (H, W, C) and dtype uint8:
    original_bits = H * W * C * 8
    
    For grayscale (H, W):
    original_bits = H * W * 8
    """
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    
    bits_per_element = 8  # Assuming uint8
    if image.dtype == np.uint16:
        bits_per_element = 16
    elif image.dtype == np.float32:
        bits_per_element = 32
    elif image.dtype == np.float64:
        bits_per_element = 64
    
    return height * width * channels * bits_per_element


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Structural Similarity Index (simplified version)."""
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
        recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY).astype(float)
    else:
        orig_gray = original.astype(float)
        recon_gray = reconstructed.astype(float)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = cv2.GaussianBlur(orig_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(recon_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(orig_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(recon_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig_gray * recon_gray, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


@dataclass
class CompressionResult:
    """Result of region-based compression"""
    original_size: int          # Original size in bits
    compressed_size: int        # Compressed size in bits
    regions: Dict[int, RegionRuns]  # region_id -> compact bbox+row-runs
    encoded_data: bytes         # All encoded data
    compression_ratio: float    # original_size / compressed_size
    computation_time: float     # Time in seconds
    num_regions: int            # Number of regions found
    image_shape: Tuple[int, ...]  # Original image shape
    image_dtype: str            # Original image dtype
    is_lossless: bool           # Always False for this method
    psnr: float                 # Peak Signal-to-Noise Ratio (dB)
    ssim: float                 # Structural Similarity Index


def _iter_region_pixels(image: np.ndarray, region: RegionRuns):
    """Yield (y, x) pixel coordinates for a region in scan order."""
    y0, x0, h, w = region.bbox
    for dy, row_runs in region.runs:
        y = y0 + dy
        for rx, length in row_runs:
            x_start = x0 + rx
            for x in range(x_start, x_start + length):
                yield y, x


class RegionBasedCompressor:
    """
    Main class for region-based lossy image compression.
    
    This compressor:
    1. Segments the image into regions based on pixel similarity
    2. Represents each region by a single color value (mean or mode)
    3. Encodes region boundaries and color values
    
    IMPORTANT: This is LOSSY compression. Pixels within a region are
    replaced by the region's representative color.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, 
                 use_mean_color: bool = True,
                 min_region_size: int = 1,
                 adaptive_segmentation: bool = True):
        """
        Args:
            similarity_threshold: Threshold for merging pixels (0-1, higher = more regions)
            use_mean_color: If True, use mean color; if False, use mode (most common)
            min_region_size: Minimum pixels per region
        """
        self.segmenter = FastRegionSegmenter()
        self.similarity_threshold = similarity_threshold
        self.use_mean_color = use_mean_color
        self.min_region_size = min_region_size
        self.adaptive_segmentation = adaptive_segmentation
        
        self.image_shape = None
        self.regions = None
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """
        Compress image using region-based lossy compression.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            CompressionResult object with compression metrics
        """
        start_time = time.time()
        self.image_shape = image.shape
        
        # Calculate correct original size
        original_size = calculate_original_size_bits(image)
        
        # Step 1: Convert image to graph
        _labels, regions = self.segmenter.segment(
            image,
            similarity_threshold=self.similarity_threshold,
            min_region_size=self.min_region_size,
            adaptive=self.adaptive_segmentation,
        )
        self.regions = regions
        
        # Step 3: Calculate representative color for each region
        region_colors = {}
        for region_id, node_list in regions.items():
            # Collect pixels for this region
            pixels = []
            for y, x in _iter_region_pixels(image, node_list):
                if image.ndim == 2:
                    pixels.append(int(image[y, x]))
                else:
                    pixels.append(tuple(int(v) for v in image[y, x]))
            
            if self.use_mean_color:
                # Use mean color
                if isinstance(pixels[0], tuple):
                    mean_color = tuple(int(np.mean([p[c] for p in pixels])) for c in range(len(pixels[0])))
                else:
                    mean_color = int(np.mean(pixels))
                region_colors[region_id] = mean_color
            else:
                # Use mode (most common)
                counter = Counter(pixels)
                region_colors[region_id] = counter.most_common(1)[0][0]
        
        # Step 4: Encode the data
        encoded_data = self._encode_regions(image.shape, regions, region_colors)
        
        # Step 5: Calculate compressed size (actual bytes * 8)
        compressed_size = len(encoded_data) * 8
        
        # Step 6: Reconstruct for quality metrics
        reconstructed = self._reconstruct_image(image.shape, regions, region_colors)
        
        # Step 7: Calculate quality metrics
        psnr = calculate_psnr(image, reconstructed)
        ssim = calculate_ssim(image, reconstructed)
        
        computation_time = time.time() - start_time
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            regions=regions,
            encoded_data=encoded_data,
            compression_ratio=compression_ratio,
            computation_time=computation_time,
            num_regions=len(regions),
            image_shape=image.shape,
            image_dtype=str(image.dtype),
            is_lossless=False,
            psnr=psnr,
            ssim=ssim
        )
    
    def _encode_regions(self, image_shape: Tuple, regions: Dict[int, RegionRuns], 
                        region_colors: Dict[int, Any]) -> bytes:
        """
        Encode regions and colors into bytes.
        
        Format (compact, spatially aware):
        - Header: shape info
        - Channels indicator (1 byte)
        - Number of regions (4 bytes)
        - For each region:
            - Region ID (4 bytes)
            - Color value (C bytes)
            - bbox: y0,x0,h,w (u16 each)
            - number of rows-with-runs (u16)
            - for each row: dy (u16), num_runs (u16), then runs: (x_start (u16), len (u16))*
        - Entire stream compressed with zlib
        """
        data = bytearray()
        
        # Header: image shape
        if len(image_shape) == 3:
            height, width, channels = image_shape
            data.extend(struct.pack('>III', height, width, channels))
        else:
            height, width = image_shape
            channels = 1
            data.extend(struct.pack('>II', height, width))
        
        # Number of channels indicator
        data.extend(struct.pack('>B', channels))
        
        # Number of regions
        data.extend(struct.pack('>I', len(regions)))
        
        # Encode each region
        for region_id, region in regions.items():
            # Region ID
            data.extend(struct.pack('>I', region_id))
            
            # Color value
            color = region_colors[region_id]
            if isinstance(color, tuple):
                for c in color:
                    data.extend(struct.pack('>B', c))
            else:
                data.extend(struct.pack('>B', color))

            # bbox + runs
            y0, x0, rh, rw = region.bbox
            if y0 > 65535 or x0 > 65535 or rh > 65535 or rw > 65535:
                raise ValueError("Image too large for u16 bbox encoding")
            data.extend(struct.pack('>HHHH', y0, x0, rh, rw))

            rows = region.runs
            if len(rows) > 65535:
                raise ValueError("Too many rows for u16 encoding")
            data.extend(struct.pack('>H', len(rows)))
            for dy, row_runs in rows:
                if dy > 65535:
                    raise ValueError("dy too large for u16 encoding")
                if len(row_runs) > 65535:
                    raise ValueError("Too many runs for u16 encoding")
                data.extend(struct.pack('>HH', dy, len(row_runs)))
                for rx, length in row_runs:
                    if rx > 65535 or length > 65535:
                        raise ValueError("run values too large for u16 encoding")
                    data.extend(struct.pack('>HH', rx, length))
        
        # Compress with zlib
        compressed = zlib.compress(bytes(data), level=9)
        return compressed
    
    def _reconstruct_image(self, image_shape: Tuple, regions: Dict[int, RegionRuns], 
                          region_colors: Dict[int, Any]) -> np.ndarray:
        """Reconstruct image from regions and colors"""
        if len(image_shape) == 3:
            height, width, channels = image_shape
            reconstructed = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            height, width = image_shape
            reconstructed = np.zeros((height, width), dtype=np.uint8)
        
        for region_id, region in regions.items():
            color = region_colors[region_id]
            y0, x0, _, _ = region.bbox
            for dy, row_runs in region.runs:
                y = y0 + dy
                for rx, length in row_runs:
                    x_start = x0 + rx
                    x_end = x_start + length
                    if isinstance(color, tuple):
                        reconstructed[y, x_start:x_end] = color
                    else:
                        reconstructed[y, x_start:x_end] = color
        
        return reconstructed
    
    def decompress(self, compression_result: CompressionResult) -> np.ndarray:
        """
        Decompress image from compression result.
        
        Note: This returns the LOSSY reconstructed image, not the original.
        """
        # Decompress data
        data = zlib.decompress(compression_result.encoded_data)
        stream = iter(data)
        
        def read_bytes(n):
            return bytes(next(stream) for _ in range(n))
        
        # Read header
        image_shape = compression_result.image_shape
        if len(image_shape) == 3:
            height, width, channels = struct.unpack('>III', read_bytes(12))
        else:
            height, width = struct.unpack('>II', read_bytes(8))
            channels = 1
        
        # Read channels indicator
        channels_check = struct.unpack('>B', read_bytes(1))[0]
        
        # Read number of regions
        num_regions = struct.unpack('>I', read_bytes(4))[0]
        
        # Create output image
        if channels > 1:
            reconstructed = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            reconstructed = np.zeros((height, width), dtype=np.uint8)
        
        # Read and apply each region
        for _ in range(num_regions):
            # Region ID
            region_id = struct.unpack('>I', read_bytes(4))[0]
            
            # Color value
            if channels > 1:
                color = tuple(struct.unpack('>B', read_bytes(1))[0] for _ in range(channels))
            else:
                color = struct.unpack('>B', read_bytes(1))[0]

            # bbox
            y0, x0, rh, rw = struct.unpack('>HHHH', read_bytes(8))

            # rows + runs
            num_rows = struct.unpack('>H', read_bytes(2))[0]
            for _ in range(num_rows):
                dy, num_runs = struct.unpack('>HH', read_bytes(4))
                y = y0 + dy
                for _ in range(num_runs):
                    rx, length = struct.unpack('>HH', read_bytes(4))
                    x_start = x0 + rx
                    x_end = x_start + length
                    if y < height:
                        reconstructed[y, x_start:x_end] = color
        
        return reconstructed
    
    def save_compressed(self, compression_result: CompressionResult, filename: str):
        """Save compressed representation to file"""
        with open(filename, 'wb') as f:
            pickle.dump(compression_result, f)
    
    def load_compressed(self, filename: str) -> CompressionResult:
        """Load compressed representation from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def visualize_segmentation(self, image: np.ndarray, regions: Dict[int, List[int]], 
                              save_path: str = None) -> np.ndarray:
        """
        Visualize region segmentation with random colors.
        
        Args:
            image: Original image
            regions: Region dictionary
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            segmentation = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            segmentation = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random colors for each region
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(50, 255, (len(regions) + 1, 3))
        
        for region_id, region in regions.items():
            color = colors[region_id % len(colors)]
            y0, x0, _, _ = region.bbox
            for dy, row_runs in region.runs:
                y = y0 + dy
                for rx, length in row_runs:
                    x_start = x0 + rx
                    x_end = x_start + length
                    segmentation[y, x_start:x_end] = color
        
        if save_path:
            cv2.imwrite(save_path, segmentation)
        
        return segmentation


# Alias for backward compatibility
MulticutCompressor = RegionBasedCompressor


def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    import matplotlib.pyplot as plt
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.rcParams["axes.unicode_minus"] = False
