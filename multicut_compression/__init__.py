"""
Multicut Image Compression Package

This package provides both LOSSY and LOSSLESS image compression using
graph-based region segmentation.

Methods:
- LosslessMulticutCompressor: LOSSLESS using Huffman coding of all pixels
- RegionBasedCompressor: LOSSY using representative colors per region
"""

from .multicut_compressor import (
    RegionBasedCompressor,
    MulticutCompressor,  # Alias for backward compatibility
    CompressionResult,
    calculate_psnr,
    calculate_ssim,
    calculate_original_size_bits
)
from .lossless_multicut import (
    LosslessMulticutCompressor,
    LosslessCompressionResult,
    verify_lossless
)
from .benchmark import CompressionBenchmark
from .dataset_loader import DatasetLoader

__version__ = "0.3.0"
__all__ = [
    # Lossless
    "LosslessMulticutCompressor",
    "LosslessCompressionResult",
    "verify_lossless",
    # Lossy
    "RegionBasedCompressor",
    "MulticutCompressor",
    "CompressionResult", 
    # Utilities
    "CompressionBenchmark",
    "DatasetLoader",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_original_size_bits"
]







