"""
Benchmark Dataset Loader for Image Compression Evaluation

This module provides functionality to load standard benchmark datasets
for image compression evaluation, including Kodak, USC-SIPI, and Waterloo datasets.


"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import urllib.request
import zipfile
import shutil
import warnings

warnings.filterwarnings('ignore')


class DatasetLoader:
    """Loader for benchmark image compression datasets"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory to store benchmark datasets. If None, uses
                the repo's code/benchmark_data folder.
        """
        if data_dir is None:
            repo_code_dir = Path(__file__).resolve().parents[1]
            data_dir = repo_code_dir / "benchmark_data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset paths (legacy naming + local benchmark_data naming)
        self.kodak_dir = self.data_dir / "kodak"
        self.photo_kodak_dir = self.data_dir / "photo_kodak"
        self.usc_sipi_dir = self.data_dir / "usc_sipi"
        self.waterloo_dir = self.data_dir / "waterloo"
        
        # Local benchmark dataset folders (from benchmark_data)
        self.local_dataset_descriptions = {
            "photo_kodak": "Kodak Lossless True Color Image Suite (24 images)",
            "photo_tecnick": "Tecnick natural image dataset",
            "photo_wikipedia": "Wikipedia photo set",
            "textures_photo": "Photographic textures",
            "textures_plants": "Plant textures",
            "textures_pk": "Textures PK",
            "textures_pk01": "Textures PK01",
            "textures_pk02": "Textures PK02",
            "screenshot_web": "Web screenshots",
            "screenshot_game": "Game screenshots",
            "icon_512": "Icon set 512px",
            "icon_64": "Icon set 64px",
            "pngimg": "PNG image collection"
        }
        self.preferred_dataset_order = [
            "photo_kodak",
            "kodak",
            "photo_tecnick",
            "photo_wikipedia",
            "textures_photo",
            "textures_plants",
            "screenshot_web",
            "screenshot_game",
            "icon_512",
            "icon_64",
            "pngimg",
            "textures_pk",
            "textures_pk01",
            "textures_pk02"
        ]
    
    def load_kodak_dataset(self, max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Load Kodak Lossless True Color Image Suite
        
        The Kodak suite comprises 24 lossless, true-color images (24-bit RGB),
        originally 768×512 or 512×768 pixels in PNG format.
        
        Args:
            max_images: Maximum number of images to load (None = all 24)
            
        Returns:
            List of (filename, image_array) tuples
        """
        images = []
        kodak_dir = self.photo_kodak_dir if self.photo_kodak_dir.exists() else self.kodak_dir
        
        # Check if dataset exists
        if not kodak_dir.exists():
            print(f"⚠️  Kodak dataset not found at {kodak_dir}")
            print("   Please download the Kodak dataset from: https://r0k.us/graphics/kodak/")
            print("   Or use load_images_from_directory() to load from a custom path")
            return images
        
        # Load PNG images
        png_files = sorted(kodak_dir.glob("*.png"))
        if not png_files:
            # Try alternative naming
            png_files = sorted(kodak_dir.glob("kodim*.png"))
        
        if max_images:
            png_files = png_files[:max_images]
        
        for png_file in png_files:
            try:
                image = cv2.imread(str(png_file))
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append((png_file.name, image))
            except Exception as e:
                print(f"⚠️  Error loading {png_file.name}: {e}")
        
        print(f"✅ Loaded {len(images)} Kodak images")
        return images
    
    def load_usc_sipi_dataset(self, volume: str = "misc", max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Load USC-SIPI Image Database
        
        Args:
            volume: Volume to load ('textures', 'aerials', 'misc', 'sequences')
            max_images: Maximum number of images to load
            
        Returns:
            List of (filename, image_array) tuples
        """
        images = []
        volume_dir = self.usc_sipi_dir / volume
        
        if not volume_dir.exists():
            print(f"⚠️  USC-SIPI {volume} volume not found at {volume_dir}")
            print("   Please download from: https://sipi.usc.edu/database/")
            return images
        
        # Load TIFF images
        tiff_files = sorted(volume_dir.glob("*.tiff")) + sorted(volume_dir.glob("*.tif"))
        if max_images:
            tiff_files = tiff_files[:max_images]
        
        for tiff_file in tiff_files:
            try:
                image = cv2.imread(str(tiff_file), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    if len(image.shape) == 3:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append((tiff_file.name, image))
            except Exception as e:
                print(f"⚠️  Error loading {tiff_file.name}: {e}")
        
        print(f"✅ Loaded {len(images)} USC-SIPI {volume} images")
        return images
    
    def load_waterloo_dataset(self, max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Load Waterloo benchmark dataset (8 images)
        
        Returns:
            List of (filename, image_array) tuples
        """
        images = []
        
        if not self.waterloo_dir.exists():
            print(f"⚠️  Waterloo dataset not found at {self.waterloo_dir}")
            print("   Please download from: http://www.researchandtechnology.net/pcif/waterloo_benchmarks.php")
            return images
        
        # Load images (PNG or BMP)
        image_files = sorted(self.waterloo_dir.glob("*.png")) + sorted(self.waterloo_dir.glob("*.bmp"))
        if max_images:
            image_files = image_files[:max_images]
        
        for img_file in image_files:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append((img_file.name, image))
            except Exception as e:
                print(f"⚠️  Error loading {img_file.name}: {e}")
        
        print(f"✅ Loaded {len(images)} Waterloo images")
        return images
    
    def load_images_from_directory(self, directory: str, pattern: str = "*.png", 
                                   max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Load images from a custom directory
        
        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.png", "*.jpg")
            max_images: Maximum number of images to load
            
        Returns:
            List of (filename, image_array) tuples
        """
        images = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {directory}")
            return images
        
        image_files = sorted(dir_path.glob(pattern))
        if max_images:
            image_files = image_files[:max_images]
        
        for img_file in image_files:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append((img_file.name, image))
            except Exception as e:
                print(f"⚠️  Error loading {img_file.name}: {e}")
        
        print(f"✅ Loaded {len(images)} images from {directory}")
        return images
    
    def list_local_datasets(self) -> List[Dict[str, Any]]:
        """List locally available datasets under benchmark_data."""
        datasets = []
        if not self.data_dir.exists():
            return datasets
        
        for item in sorted(self.data_dir.iterdir()):
            if not item.is_dir():
                continue
            image_files = self._get_image_files(item)
            if image_files:
                datasets.append({
                    "name": item.name,
                    "path": str(item),
                    "count": len(image_files),
                    "description": self.local_dataset_descriptions.get(item.name, "Local dataset")
                })
        return datasets

    def iter_all_local_datasets(
        self,
        max_images_per_dataset: Optional[int] = None,
    ) -> List[Tuple[str, List[Tuple[str, np.ndarray]]]]:
        """
        Load images from every dataset folder under benchmark_data.

        Returns:
            List of (dataset_name, [(filename, image), ...])
        """
        all_sets: List[Tuple[str, List[Tuple[str, np.ndarray]]]] = []
        datasets = self.list_local_datasets()
        for ds in datasets:
            name = ds["name"]
            images = self.load_local_dataset(name, max_images=max_images_per_dataset)
            if images:
                all_sets.append((name, images))
        return all_sets
    
    def load_local_dataset(self, dataset_name: str, max_images: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Load images from a known local dataset folder."""
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return []
        
        image_files = self._get_image_files(dataset_path)
        if max_images:
            image_files = image_files[:max_images]
        
        images = []
        for img_file in image_files:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append((img_file.name, image))
            except Exception as e:
                print(f"⚠️  Error loading {img_file.name}: {e}")
        
        if images:
            print(f"✅ Loaded {len(images)} images from {dataset_name}")
        return images
    
    def load_preferred_images(self, max_images: int) -> List[Tuple[str, np.ndarray]]:
        """Load images from the first available preferred dataset."""
        # Check local benchmark_data folders
        for dataset_name in self.preferred_dataset_order:
            if dataset_name == "kodak":
                images = self.load_kodak_dataset(max_images=max_images)
            elif dataset_name == "waterloo":
                images = self.load_waterloo_dataset(max_images=max_images)
            elif dataset_name == "usc_sipi":
                images = self.load_usc_sipi_dataset(max_images=max_images)
            else:
                images = self.load_local_dataset(dataset_name, max_images=max_images)
            if images:
                return images
        return []
    
    def prepare_dataset_info(self) -> Dict[str, any]:
        """
        Get information about available datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'kodak': {
                'available': (self.photo_kodak_dir.exists() and len(list(self.photo_kodak_dir.glob("*.png"))) > 0)
                             or (self.kodak_dir.exists() and len(list(self.kodak_dir.glob("*.png"))) > 0),
                'count': len(list(self.photo_kodak_dir.glob("*.png"))) if self.photo_kodak_dir.exists()
                         else (len(list(self.kodak_dir.glob("*.png"))) if self.kodak_dir.exists() else 0),
                'path': str(self.photo_kodak_dir if self.photo_kodak_dir.exists() else self.kodak_dir),
                'description': 'Kodak Lossless True Color Image Suite (24 images)'
            },
            'usc_sipi': {
                'available': self.usc_sipi_dir.exists(),
                'volumes': [],
                'path': str(self.usc_sipi_dir),
                'description': 'USC-SIPI Image Database'
            },
            'waterloo': {
                'available': self.waterloo_dir.exists() and len(list(self.waterloo_dir.glob("*.png"))) > 0,
                'count': len(list(self.waterloo_dir.glob("*.png"))) if self.waterloo_dir.exists() else 0,
                'path': str(self.waterloo_dir),
                'description': 'Waterloo benchmark dataset (8 images)'
            },
            'local': self.list_local_datasets()
        }
        
        # Check USC-SIPI volumes
        if self.usc_sipi_dir.exists():
            for volume in ['textures', 'aerials', 'misc', 'sequences']:
                vol_dir = self.usc_sipi_dir / volume
                if vol_dir.exists():
                    count = len(list(vol_dir.glob("*.tiff"))) + len(list(vol_dir.glob("*.tif")))
                    if count > 0:
                        info['usc_sipi']['volumes'].append({
                            'name': volume,
                            'count': count
                        })
        
        return info

    def _get_image_files(self, dir_path: Path) -> List[Path]:
        """Get image files from a directory with common extensions."""
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for pattern in patterns:
            files.extend(sorted(dir_path.glob(pattern)))
        return files


def download_kodak_dataset(output_dir: str = "benchmark_data/kodak", 
                          url: str = None) -> bool:
    """
    Download Kodak dataset (helper function)
    
    Note: This is a placeholder. Users should download manually from:
    https://r0k.us/graphics/kodak/
    
    Args:
        output_dir: Output directory
        url: Optional URL (not provided by default)
        
    Returns:
        True if successful
    """
    print("⚠️  Automatic download not implemented.")
    print("   Please download Kodak dataset manually from:")
    print("   https://r0k.us/graphics/kodak/")
    print(f"   Extract to: {output_dir}")
    return False







