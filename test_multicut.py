"""
Simple Test Script for Multi-Cut Image Compression

This script demonstrates the basic functionality of the multi-cut compression pipeline
and provides a quick test of all components.
"""

import numpy as np
import cv2
import time
import os
from pathlib import Path
from typing import Tuple

# Import our modules
import sys
sys.path.append('multicut_compression')
from multicut_compression.multicut_compressor import MulticutCompressor
from multicut_compression.benchmark import CompressionBenchmark
from multicut_compression.dataset_loader import DatasetLoader

def downscale_image(image: np.ndarray, max_dimension: int = 256) -> np.ndarray:
    """Downscale images to keep runtime reasonable."""
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= max_dimension:
        return image
    scale = max_dimension / max_dim
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def load_real_test_image() -> Tuple[str, np.ndarray]:
    """Load a real benchmark image from benchmark_data."""
    dataset_loader = DatasetLoader()
    images = dataset_loader.load_preferred_images(max_images=1)
    if not images:
        raise FileNotFoundError(f"No real datasets found in {dataset_loader.data_dir}")
    filename, image = images[0]
    return filename, downscale_image(image, max_dimension=256)

def test_basic_functionality():
    """Test basic functionality of the compression pipeline"""
    
    print("Testing Multi-Cut Image Compression")
    print("=" * 50)
    
    # Load real test image
    print("Loading real test image...")
    filename, test_image = load_real_test_image()
    print(f"Image: {filename} {test_image.shape}")
    print(f"Image size: {test_image.size} pixels")
    print(f"Original size (bits): {test_image.size * 8}")
    
    # Save original image
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "test_original.png"), test_image)
    
    # Initialize compressor
    print("\nInitializing compressor...")
    compressor = MulticutCompressor()
    
    # Test compression
    print("Compressing image...")
    start_time = time.time()
    result = compressor.compress(test_image)
    compression_time = time.time() - start_time
    
    print(f"Compression time: {compression_time:.3f} seconds")
    print(f"Number of regions found: {result.num_regions}")
    print(f"Compressed size: {result.compressed_size} bits")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Space savings: {(1 - result.compressed_size/result.original_size)*100:.1f}%")
    
    # Test reconstruction
    print("\nReconstructing image...")
    reconstructed = compressor.decompress(result)
    
    # Calculate reconstruction quality
    mse = np.mean((test_image.astype(float) - reconstructed.astype(float))**2)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    print(f"Reconstruction MSE: {mse:.2f}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    
    # Check if reconstruction is perfect
    if np.array_equal(test_image, reconstructed):
        print("✓ Perfect reconstruction (lossless)")
    else:
        print("✗ Reconstruction has some differences")
    
    # Save results
    print("\nSaving results...")
    cv2.imwrite(str(output_dir / "test_reconstructed.png"), reconstructed)
    
    # Visualize segmentation
    segmentation = compressor.visualize_segmentation(
        test_image, result.regions, 
        save_path=str(output_dir / "test_segmentation.png")
    )
    
    # Save compressed data
    compressor.save_compressed(result, str(output_dir / "test_compressed.pkl"))
    
    # Test loading compressed data
    loaded_result = compressor.load_compressed(str(output_dir / "test_compressed.pkl"))
    print("✓ Compressed data saved and loaded successfully")
    
    return result

def test_benchmark_framework():
    """Test the benchmark framework"""
    
    print("\n" + "=" * 50)
    print("Testing Benchmark Framework")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = CompressionBenchmark()
    
    # Test with a small number of images
    print("Running quick benchmark (2 images)...")
    results = benchmark.run_benchmark(num_images=2)
    
    # Generate report
    report = benchmark.generate_report(results)
    print("\n" + "=" * 30 + " BENCHMARK REPORT " + "=" * 30)
    print(report)
    
    return results

def test_region_analysis():
    """Detailed analysis of regions found"""
    
    print("\n" + "=" * 50)
    print("Detailed Region Analysis")
    print("=" * 50)
    
    # Load a real test image for analysis
    _, image = load_real_test_image()
    compressor = MulticutCompressor()
    
    print("Compressing with detailed analysis...")
    result = compressor.compress(image)
    
    print(f"Total regions: {result.num_regions}")
    print(f"Regions with different sizes:")
    
    for region_id, node_list in result.regions.items():
        region_size = len(node_list)
        print(f"  Region {region_id}: {region_size} pixels")
    
    # Analyze region statistics
    region_sizes = [len(nodes) for nodes in result.regions.values()]
    if region_sizes:
        print(f"\nRegion statistics:")
        print(f"  Min size: {min(region_sizes)} pixels")
        print(f"  Max size: {max(region_sizes)} pixels")
        print(f"  Average size: {np.mean(region_sizes):.1f} pixels")
        print(f"  Standard deviation: {np.std(region_sizes):.1f} pixels")
    else:
        print("\nNo regions were found - this indicates the algorithm needs tuning")

def main():
    """Main test function"""
    
    print("Multi-Cut Image Compression - Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        basic_result = test_basic_functionality()
        
        # Test 2: Region analysis
        test_region_analysis()
        
        # Test 3: Benchmark framework (quick test)
        benchmark_results = test_benchmark_framework()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Test results saved in: {os.path.abspath('test_results')}")
        print("Generated files:")
        print("  - test_original.png: Original test image")
        print("  - test_reconstructed.png: Reconstructed image")
        print("  - test_segmentation.png: Segmentation visualization")
        print("  - test_compressed.pkl: Compressed data")
        print("  - benchmark_*.png: Benchmark visualizations")
        print("  - benchmark_*.json: Benchmark results")
        print("  - benchmark_*.txt: Benchmark report")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
