"""
Lossless Multicut+Huffman Compression - Complete Demonstration

This script demonstrates LOSSLESS image compression using:
- Graph-based region segmentation (multicut)
- Huffman coding to encode all pixel values within regions

Compared against 8 lossless methods:
- PNG, JPEG-LS, JPEG 2000, JPEG XR, WebP, BPG, FLIF (2 variants)


"""

import numpy as np
import cv2
import time
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from multicut_compression.lossless_multicut import (
    LosslessMulticutCompressor,
    verify_lossless
)
from multicut_compression.multicut_compressor import calculate_original_size_bits
from multicut_compression.benchmark import CompressionBenchmark
from multicut_compression.dataset_loader import DatasetLoader


def downscale_image(image: np.ndarray, max_dimension: int = 128) -> np.ndarray:
    """Downscale images to keep runtime reasonable.
    
    NOTE: Using 128px max for demonstration speed.
    Larger images take significantly longer due to graph processing.
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= max_dimension:
        return image
    scale = max_dimension / max_dim
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def demonstrate_lossless_pipeline():
    """Demonstrate the lossless Multicut+Huffman compression pipeline"""
    
    print("=" * 70)
    print("LOSSLESS MULTICUT+HUFFMAN COMPRESSION DEMONSTRATION")
    print("=" * 70)
    print()
    print("This method is TRULY LOSSLESS - all pixel values are preserved exactly.")
    print("Uses: Region segmentation + Huffman coding of all pixels in each region.")
    print()
    
    os.makedirs('demo_results', exist_ok=True)
    
    # Initialize lossless compressor
    compressor = LosslessMulticutCompressor(
        similarity_threshold=0.95,  # Higher = more regions = potentially better compression
        min_region_size=1
    )
    
    print("Loading benchmark image...")
    dataset_loader = DatasetLoader()
    images = dataset_loader.load_preferred_images(max_images=1)
    if not images:
        raise FileNotFoundError(f"No datasets found in {dataset_loader.data_dir}")
    
    filename, test_image = images[0]
    test_image = downscale_image(test_image, max_dimension=256)
    
    cv2.imwrite('demo_results/demo_original.png', test_image)
    print(f"Original image: {filename}")
    print(f"Shape: {test_image.shape}")
    print(f"Original size: {calculate_original_size_bits(test_image):,} bits")
    print()
    
    print("Running Lossless Multicut+Huffman Compression...")
    print("-" * 50)
    result = compressor.compress(test_image)
    
    print(f"Compression time: {result.computation_time:.3f} seconds")
    print(f"Number of regions: {result.num_regions}")
    print(f"Original size: {result.original_size:,} bits")
    print(f"Compressed size: {result.compressed_size:,} bits")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Encoding method: {result.encoding_method}")
    print()
    
    # Verify lossless
    print("Verifying LOSSLESS compression...")
    print("-" * 50)
    reconstructed = compressor.decompress(result)
    is_lossless = verify_lossless(test_image, reconstructed)
    
    if is_lossless:
        print("✅ VERIFIED: Compression is TRULY LOSSLESS!")
        print("   All pixel values match exactly.")
    else:
        print("❌ WARNING: Some pixel values differ!")
        diff = np.abs(test_image.astype(float) - reconstructed.astype(float))
        print(f"   Max difference: {diff.max()}")
        print(f"   Mean difference: {diff.mean():.4f}")
    
    cv2.imwrite('demo_results/demo_reconstructed.png', reconstructed)
    compressor.save_compressed(result, 'demo_results/demo_compressed.pkl')
    
    print()
    return result, is_lossless


def run_lossless_benchmark():
    """Run benchmark comparing Multicut+Huffman vs other lossless methods"""
    
    print()
    print("=" * 70)
    print("LOSSLESS COMPRESSION BENCHMARK")
    print("Multicut+Huffman vs Standard Lossless Methods")
    print("=" * 70)
    print()
    print("Methods compared:")
    print("  Multicut variants (LOSSLESS):")
    print("    1. Multicut-RLE (LOSSLESS)")
    print("    2. Multicut-Huffman (LOSSLESS)")
    print("    3. Multicut-Predictor+Zlib (LOSSLESS)")
    print("")
    print("  Standard lossless baselines:")
    print("    PNG, JPEG-LS (simulated), JPEG2000-Lossless, JPEGXR-Lossless, WebP-Lossless,")
    print("    BPG-Lossless, FLIF-Default, FLIF-NonInterlaced")
    print()
    
    benchmark = CompressionBenchmark('demo_results')
    
    print("Testing with 5 diverse images...")
    results = benchmark.run_benchmark(num_images=5)
    
    report = benchmark.generate_report(results)
    
    with open('demo_results/benchmark_report.txt', 'w') as f:
        f.write(report)
    
    print()
    print("Generating visualizations...")
    benchmark.create_visualizations(results)
    
    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    mh_summary = results.get('multicut_huffman_summary', {})
    if mh_summary:
        print(f"\nMulticut+Huffman (Our Method):")
        print(f"  Compression Ratio: {mh_summary.get('avg_compression_ratio', 0):.2f}x")
        print(f"  Avg Regions: {mh_summary.get('avg_num_regions', 0):.0f}")
        print(f"  Avg Time: {mh_summary.get('avg_computation_time', 0):.2f}s")
        print(f"  Truly Lossless: {'✅ Yes' if mh_summary.get('is_lossless', False) else '❌ No'}")
    
    print(f"\nOther Lossless Methods:")
    methods = ['PNG', 'JPEG-LS', 'JPEG2000-Lossless', 'WebP-Lossless', 
               'JPEGXR-Lossless', 'BPG-Lossless', 'FLIF-Default', 'FLIF-NonInterlaced']
    
    for method in methods:
        summary_key = f'{method}_summary'
        summary = results['lossless_methods'].get(summary_key)
        if summary:
            validated = "✓" if summary.get('validated', True) else "~"
            print(f"  {validated} {method}: {summary['avg_compression_ratio']:.2f}x")
    
    print()
    print("Legend: ✓ = Validated, ~ = Estimated")
    
    return results


def demonstrate_threshold_effect():
    """Show how similarity threshold affects compression"""
    
    print()
    print("=" * 70)
    print("THRESHOLD EFFECT ON LOSSLESS COMPRESSION")
    print("=" * 70)
    print()
    
    dataset_loader = DatasetLoader()
    images = dataset_loader.load_preferred_images(max_images=1)
    if not images:
        raise FileNotFoundError("No dataset found")
    
    _, test_image = images[0]
    test_image = downscale_image(test_image, max_dimension=256)
    
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.98]
    
    print(f"{'Threshold':<12} {'Regions':>10} {'Ratio':>12} {'Time':>10} {'Lossless':>10}")
    print("-" * 58)
    
    for threshold in thresholds:
        compressor = LosslessMulticutCompressor(
            similarity_threshold=threshold,
            min_region_size=1
        )
        result = compressor.compress(test_image)
        
        # Verify lossless
        reconstructed = compressor.decompress(result)
        is_lossless = verify_lossless(test_image, reconstructed)
        
        print(f"{threshold:<12.2f} {result.num_regions:>10} "
              f"{result.compression_ratio:>12.2f}x "
              f"{result.computation_time:>10.2f}s "
              f"{'✅' if is_lossless else '❌':>10}")
    
    print()
    print("Higher threshold → More regions → Different compression characteristics")
    print("All thresholds should be lossless (✅) if implementation is correct.")


def main():
    """Main demonstration function"""
    
    print()
    print("=" * 80)
    print("LOSSLESS MULTICUT+HUFFMAN COMPRESSION - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demonstration showcases TRULY LOSSLESS image compression using:")
    print("  - Graph-based region segmentation (multicut)")
    print("  - Huffman coding to encode ALL pixel values within each region")
    print()
    print("Unlike lossy methods, this preserves every pixel exactly!")
    print()
    
    try:
        # 1. Core pipeline demonstration
        print("PHASE 1: CORE LOSSLESS PIPELINE")
        print("=" * 50)
        result, is_lossless = demonstrate_lossless_pipeline()
        
        if not is_lossless:
            print("\n⚠️  WARNING: Lossless verification failed!")
            print("    There may be a bug in the implementation.")
        
        # 2. Benchmark against other lossless methods
        print()
        print("PHASE 2: LOSSLESS BENCHMARK")
        print("=" * 50)
        benchmark_results = run_lossless_benchmark()
        
        # 3. Threshold effect
        print()
        print("PHASE 3: THRESHOLD ANALYSIS")
        print("=" * 50)
        demonstrate_threshold_effect()
        
        # Final summary
        print()
        print("=" * 70)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print()
        print("Generated files in demo_results/:")
        print("  - demo_original.png: Original test image")
        print("  - demo_reconstructed.png: Reconstructed (should be IDENTICAL)")
        print("  - demo_compressed.pkl: Compressed data")
        print("  - benchmark_report.txt: Full benchmark report")
        print("  - benchmark_visualization.png: Comparison charts")
        print("  - detailed_comparison.png: Detailed analysis")
        print()
        print("KEY POINTS:")
        print("  1. Multicut+Huffman is TRULY LOSSLESS")
        print("  2. Exploits spatial redundancy via region segmentation")
        print("  3. Huffman codes similar pixels efficiently")
        print("  4. Compare fairly with other lossless methods")
        
        return True
        
    except Exception as e:
        print(f"\nDEMONSTRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
