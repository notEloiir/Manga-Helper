"""OCR module for text detection and extraction from manga images."""

import os
import sys
from math import hypot
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


def _as_polygons(bboxes):
    """Normalize one polygon or a list of polygons to a list of polygons."""
    if len(bboxes) == 0:
        return []
    if all(len(point) == 2 for point in bboxes):
        return [bboxes]
    return bboxes


def _all_points(bboxes):
    return [point for polygon in _as_polygons(bboxes) for point in polygon]


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point, start, end):
    return (
        _orientation(start, end, point) == 0
        and min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
    )


def _segments_intersect(start1, end1, start2, end2):
    orientations = (
        _orientation(start1, end1, start2),
        _orientation(start1, end1, end2),
        _orientation(start2, end2, start1),
        _orientation(start2, end2, end1),
    )
    if orientations[0] == 0 and _point_on_segment(start2, start1, end1):
        return True
    if orientations[1] == 0 and _point_on_segment(end2, start1, end1):
        return True
    if orientations[2] == 0 and _point_on_segment(start1, start2, end2):
        return True
    if orientations[3] == 0 and _point_on_segment(end1, start2, end2):
        return True
    return (orientations[0] > 0) != (orientations[1] > 0) and (orientations[2] > 0) != (orientations[3] > 0)


def _point_in_polygon(point, polygon):
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            x_at_y = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
            if point[0] < x_at_y:
                inside = not inside
    return inside


def _segment_distance(start1, end1, start2, end2):
    if _segments_intersect(start1, end1, start2, end2):
        return 0

    def point_segment_distance(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if dx == 0 and dy == 0:
            return hypot(point[0] - start[0], point[1] - start[1])
        factor = max(0, min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
        return hypot(point[0] - (start[0] + factor * dx), point[1] - (start[1] + factor * dy))

    return min(
        point_segment_distance(start1, start2, end2),
        point_segment_distance(end1, start2, end2),
        point_segment_distance(start2, start1, end1),
        point_segment_distance(end2, start1, end1),
    )


def _polygons_overlap_or_close(polygon1, polygon2, threshold):
    for index, start1 in enumerate(polygon1):
        end1 = polygon1[(index + 1) % len(polygon1)]
        for index, start2 in enumerate(polygon2):
            end2 = polygon2[(index + 1) % len(polygon2)]
            if _segment_distance(start1, end1, start2, end2) <= threshold:
                return True
    if _point_in_polygon(polygon1[0], polygon2) or _point_in_polygon(polygon2[0], polygon1):
        return True
    return False


def _bboxes_overlap_or_close(bbox1, bbox2, threshold=0):
    """Check overlap or proximity using the original polygon geometry."""
    return any(
        _polygons_overlap_or_close(polygon1, polygon2, threshold)
        for polygon1 in _as_polygons(bbox1)
        for polygon2 in _as_polygons(bbox2)
    )


def _merge_bboxes(bbox1, bbox2):
    """Combine original polygons into one merged detection."""
    return _as_polygons(bbox1) + _as_polygons(bbox2)


def detect_text_regions(image_path, image: Image.Image):
    """Detect text regions in the image using EasyOCR.
    
    Returns list of tuples: (bboxes, text, confidence), where bboxes is a list
    of the original detected polygons belonging to the merged region.
    """
    reader = get_easyocr_reader()
    # rotation_info should help with vertical text detection, but it doesn't.
    results = reader.readtext(str(image_path),
        detail=1,
        add_margin=0.0,
        text_threshold=0.1,
        link_threshold=0.1,
        low_text=0.3,
    )
    results = [([bbox], text, confidence) for bbox, text, confidence in results]

    # Merge overlapping or very close bounding boxes to reduce duplicates
    # paragraph=True does this automatically, but we want to handle it manually for smaller box to clean
    threshold = max(image.size) * 0.004  # .4% of the image size as threshold
    found_overlapping = True
    while found_overlapping:
        found_overlapping = False
        for i in range(len(results)):
            bbox1, text1, conf1 = results[i]
            for j in range(i + 1, len(results)):
                bbox2, text2, conf2 = results[j]
                # Check if bounding boxes overlap or are very close
                if _bboxes_overlap_or_close(bbox1, bbox2, threshold):
                    # Merge the two boxes
                    merged_bbox = _merge_bboxes(bbox1, bbox2)
                    merged_text = f"{text1} {text2}".strip()
                    merged_conf = max(conf1, conf2)
                    results.append((merged_bbox, merged_text, merged_conf))
                    results.pop(j)
                    results.pop(i)
                    found_overlapping = True
                    break
            if found_overlapping:
                break
    
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
    points = _all_points(bbox)
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
        y_min = min(point[1] for point in _all_points(bbox))
        
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
        row_regions.sort(key=lambda r: min(p[0] for p in _all_points(r[0])), reverse=True)
        sorted_regions.extend(row_regions)
    
    return sorted_regions


def extract_text(image: Image.Image, text_regions: list):
    """Extract Japanese text from detected regions using MangaOCR.
    
    Properly handles both vertical and horizontal Japanese text.
    
    Returns list of strings in reading order.
    """
    
    texts = []
    for region in text_regions:
        bbox = _all_points(region[0])
        
        # Use MangaOCR to properly extract text from this region
        text = extract_text_from_region(image, bbox)
        texts.append(text)
    
    return texts
