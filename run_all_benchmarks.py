"""
Run benchmarks across ALL datasets in code/benchmark_data.

This script is designed to satisfy the PDF evaluation requirement:
"evaluation on a wide variety of images".

It runs:
1) Our multicut-based variants (multiple b(R) encoders)
2) Standard lossless baselines (PNG/WebP + optional external tools if installed)

Outputs:
- code/demo_results_all/<dataset_name>/benchmark_results.json
- code/demo_results_all/<dataset_name>/benchmark_report.txt
- code/demo_results_all/<dataset_name>/*.png plots

Usage:
  source code/.venv/bin/activate
  python code/run_all_benchmarks.py

Optional args:
  --max-images-per-dataset 10
  --max-dimension 0     (0 means full resolution)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multicut_compression.dataset_loader import DatasetLoader
from multicut_compression.benchmark import CompressionBenchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images-per-dataset", type=int, default=10)
    ap.add_argument("--max-dimension", type=int, default=0, help="0 = full resolution")
    args = ap.parse_args()

    out_root = Path(__file__).resolve().parent / "demo_results_all"
    out_root.mkdir(exist_ok=True)

    dl = DatasetLoader()
    datasets = dl.list_local_datasets()
    if not datasets:
        print("No datasets found under code/benchmark_data.")
        return 1

    for ds in datasets:
        name = ds["name"]
        print("\n" + "=" * 80)
        print(f"DATASET: {name} ({ds.get('count', '?')} files)")
        print("=" * 80)

        images = dl.load_local_dataset(name, max_images=args.max_images_per_dataset)
        if not images:
            print("  (skipping: no images loaded)")
            continue

        ds_out = out_root / name
        bench = CompressionBenchmark(output_dir=str(ds_out))

        # Configure full-res vs downscale
        max_dim = None if args.max_dimension == 0 else args.max_dimension
        results = bench.run_benchmark_on_images(images, max_dimension=max_dim)
        report = bench.generate_report(results)
        (ds_out / "benchmark_report.txt").write_text(report)
        bench.create_visualizations(results)

        print(f"Saved: {ds_out}")

    print("\nAll datasets processed.")
    print(f"Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

