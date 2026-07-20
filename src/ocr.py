"""OCR module for text detection and extraction from manga images."""

import sys
import numpy as np
from PIL import Image
import easyocr
from manga_ocr import MangaOcr


# Global readers (initialized once for performance)
_easyocr_reader = None
_manga_ocr = None


def get_easyocr_reader():
    """Get or initialize the EasyOCR reader for text detection."""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("Initializing text detection model...", file=sys.stderr)
        _easyocr_reader = easyocr.Reader(['ja'], gpu=False)
    return _easyocr_reader


def get_manga_ocr():
    """Get or initialize the MangaOCR reader for vertical text recognition."""
    global _manga_ocr
    if _manga_ocr is None:
        print("Initializing manga OCR model...", file=sys.stderr)
        _manga_ocr = MangaOcr()
    return _manga_ocr


def detect_text_regions(image_path):
    """Detect text regions in the image using EasyOCR.
    
    Returns list of tuples: (bbox, text, confidence)
    where bbox is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    reader = get_easyocr_reader()
    results = reader.readtext(str(image_path), detail=1)
    
    return results


def extract_text_from_region(image, bbox):
    """Extract text from a specific region using MangaOCR.
    
    Handles both vertical and horizontal Japanese text correctly.
    """
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
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


def extract_text(image_path, text_regions):
    """Extract Japanese text from detected regions using MangaOCR.
    
    Properly handles both vertical and horizontal Japanese text.
    
    Returns list of strings in reading order.
    """
    # Open the full image
    image = Image.open(image_path)
    img_array = np.array(image)
    
    texts = []
    for region in text_regions:
        bbox = region[0]
        
        # Use MangaOCR to properly extract text from this region
        text = extract_text_from_region(img_array, bbox)
        texts.append(text)
    
    return texts
