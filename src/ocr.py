"""OCR module for text detection and extraction from manga images."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import easyocr
from huggingface_hub import snapshot_download


# Global readers (initialized once for performance)
_easyocr_reader = None
_manga_ocr = None


# Local cache directories for downloaded model assets
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EASYOCR_CACHE_DIR = _PROJECT_ROOT / ".cache" / "easyocr"
_MANGA_OCR_CACHE_DIR = _PROJECT_ROOT / ".cache" / "manga_ocr"
_EASYOCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MANGA_OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Prefer using the existing local Hugging Face hub cache when available
_HUGGINGFACE_HUB_CACHE = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
_HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)

# Force transformers/hf hub environment variables to respect the local cache path
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MANGA_OCR_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HUGGINGFACE_HUB_CACHE))
os.environ.setdefault("HF_HOME", str(_HUGGINGFACE_HUB_CACHE.parent))

from manga_ocr import MangaOcr


def _get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_easyocr_reader():
    """Get or initialize the EasyOCR reader for text detection."""
    global _easyocr_reader
    if _easyocr_reader is None:
        device_name = _get_device_name()
        use_gpu = device_name == "cuda"
        print(f"Initializing text detection model on {device_name.upper()}...", file=sys.stderr)
        _easyocr_reader = easyocr.Reader(
            ['ja'],
            gpu=use_gpu,
            model_storage_directory=str(_EASYOCR_CACHE_DIR),
            download_enabled=True,
        )
    return _easyocr_reader


def get_manga_ocr():
    """Get or initialize the MangaOCR reader for vertical text recognition."""
    global _manga_ocr
    if _manga_ocr is None:
        device_name = _get_device_name()
        force_cpu = device_name == "cpu"
        print(f"Initializing manga OCR model on {device_name.upper()}...", file=sys.stderr)

        model_name = "kha-white/manga-ocr-base"
        local_model_dir = None

        try:
            cached_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(_HUGGINGFACE_HUB_CACHE),
                local_files_only=True,
            )
            local_model_dir = Path(cached_path)
            print(f"Using local MangaOCR cache at: {local_model_dir}", file=sys.stderr)
        except Exception as e:
            print(f"MangaOCR local cache lookup failed: {e}", file=sys.stderr)
            local_model_dir = None

        if local_model_dir and local_model_dir.exists():
            _manga_ocr = MangaOcr(pretrained_model_name_or_path=str(local_model_dir), force_cpu=force_cpu)
        else:
            print("Falling back to remote MangaOCR repo URL. This may require internet access.", file=sys.stderr)
            _manga_ocr = MangaOcr(force_cpu=force_cpu)

    return _manga_ocr


def detect_text_regions(image_path):
    """Detect text regions in the image using EasyOCR.
    
    Returns list of tuples: (bbox, text, confidence)
    where bbox is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    reader = get_easyocr_reader()
    results = reader.readtext(str(image_path), detail=1)
    
    return results


def extract_text_from_region(image: Image.Image, bbox: list):
    """Extract text from a specific region using MangaOCR.
    
    Handles both vertical and horizontal Japanese text correctly.
    """
    try:
        img_array = np.array(image)
        
        # Get bounding box coordinates
        points = np.array(bbox, dtype=np.int32)
        
        # Create a cropped image from the region
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add small padding
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_array.shape[1], x_max + padding)
        y_max = min(img_array.shape[0], y_max + padding)
        
        cropped = img_array[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return ""
        
        # Convert back to PIL for MangaOCR
        cropped_pil = Image.fromarray(cropped) if len(cropped.shape) == 3 else Image.fromarray(cropped, mode='L')
        
        # Use MangaOCR for text recognition (handles vertical text)
        mocr = get_manga_ocr()
        text = mocr(cropped_pil)
        
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting text from region: {e}", file=sys.stderr)
        return ""


def is_vertical_text(bbox):
    """Determine if a text region is likely vertical (right-to-left, top-to-bottom).
    
    Checks if bounding box height is significantly greater than width.
    """
    points = bbox
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    # If height is significantly greater than width, it's likely vertical text
    return height > width * 1.2


def sort_text_regions(text_regions):
    """Sort detected text regions in manga reading order (top to bottom, right to left).
    
    Regions are sorted by:
    1. Y-coordinate (top to bottom)
    2. X-coordinate (right to left within same y-level)
    """
    # Group regions by approximate row (y-coordinate)
    # Allow some tolerance for regions on the same line
    row_tolerance = 30
    
    rows = {}
    for region in text_regions:
        bbox = region[0]
        # Get the top-y coordinate of the region
        y_min = min(point[1] for point in bbox)
        
        # Find which row this belongs to
        row_key = None
        for existing_y in rows.keys():
            if abs(y_min - existing_y) < row_tolerance:
                row_key = existing_y
                break
        
        if row_key is None:
            row_key = y_min
        
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(region)
    
    # Sort rows by y-coordinate (top to bottom)
    sorted_regions = []
    for y_key in sorted(rows.keys()):
        # Within each row, sort by x-coordinate (right to left)
        row_regions = rows[y_key]
        row_regions.sort(key=lambda r: min(p[0] for p in r[0]), reverse=True)
        sorted_regions.extend(row_regions)
    
    return sorted_regions


def extract_text(image: Image.Image, text_regions: list):
    """Extract Japanese text from detected regions using MangaOCR.
    
    Properly handles both vertical and horizontal Japanese text.
    
    Returns list of strings in reading order.
    """
    
    texts = []
    for region in text_regions:
        bbox = region[0]
        
        # Use MangaOCR to properly extract text from this region
        text = extract_text_from_region(image, bbox)
        texts.append(text)
    
    return texts
