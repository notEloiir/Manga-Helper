#!/usr/bin/env python3
"""CLI tool for manga translation: detect Japanese text, extract via OCR, and white out text areas."""

import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import easyocr
from manga_ocr import MangaOcr
import numpy as np
from kanjiconv import KanjiConv
from deep_translator import GoogleTranslator


# Global readers (initialized once for performance)
_easyocr_reader = None
_manga_ocr = None
_kanjiconv = None

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


def convert_to_hiragana(text):
    """Convert Japanese text (kanji + hiragana) to hiragana phonetics.
    
    Uses kanjiconv to convert all characters to hiragana.
    """
    global _kanjiconv
    if _kanjiconv is None:
        _kanjiconv = KanjiConv(separator="", use_unidic=True)
    _kanjiconv.separator = ""  # No separator for hiragana

    try:
        return _kanjiconv.to_hiragana(text)
    except Exception as e:
        print(f"Error converting to hiragana: {e}", file=sys.stderr)
        return text


def convert_to_romaji(text):
    """Convert Japanese text to romaji (Latin script).
    
    Uses kanjiconv to convert all characters to romaji.
    """
    global _kanjiconv
    if _kanjiconv is None:
        _kanjiconv = KanjiConv(separator=" ", use_unidic=True)
    _kanjiconv.separator = " "  # Ensure space between words for romaji

    try:
        return _kanjiconv.to_roman(text)
    except Exception as e:
        print(f"Error converting to romaji: {e}", file=sys.stderr)
        return text


def translate_with_context(texts):
    """Translate Japanese text to English using full context.
    
    Combines all text snippets for better context-aware translation.
    Returns list of translations in same order as input.
    """
    if not texts or all(not t for t in texts):
        return [""] * len(texts)
    
    try:
        translator = GoogleTranslator(source_language='ja', target_language='en')
        
        # Create context string from all non-empty texts
        context = " ".join(t for t in texts if t.strip())
        
        # If context is too short, translate individual texts
        if len(context) < 20:
            return [translator.translate(t) if t.strip() else "" for t in texts]
        
        # For each text, include surrounding context for better translation
        translations = []
        for i, text in enumerate(texts):
            if not text.strip():
                translations.append("")
                continue
            
            # Build context with this snippet highlighted
            before = " ".join(texts[max(0, i-1):i])
            after = " ".join(texts[i+1:min(len(texts), i+2)])
            full_context = f"{before} {text} {after}".strip()
            
            # Translate with context
            try:
                translation = translator.translate(full_context)
                # Extract just the translation of the main text
                # This is a heuristic - we translate with context but the user sees full translation
                translations.append(translation)
            except Exception as e:
                print(f"Error translating text: {e}", file=sys.stderr)
                translations.append("")
        
        return translations
    
    except Exception as e:
        print(f"Error initializing translator: {e}", file=sys.stderr)
        return [""] * len(texts)


def whiteout_text_areas(image_path, output_path, text_regions, opacity=0.8):
    """White out detected text areas with specified opacity.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        text_regions: List of detected text regions with bboxes
        opacity: Opacity of white overlay (0.0-1.0), where 1.0 is fully opaque
    """
    # Open image and convert to RGBA for transparency support
    image = Image.open(image_path).convert('RGBA')
    width, height = image.size
    
    # Create overlay layer
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Calculate alpha value (0-255 where 255 is fully opaque)
    alpha = int(255 * opacity)
    white_with_opacity = (255, 255, 255, alpha)
    
    # Draw white rectangles over each text region
    for region in text_regions:
        bbox = region[0]
        
        # Convert bbox points to polygon coordinates
        # bbox is already a list of (x, y) tuples
        polygon = [(int(p[0]), int(p[1])) for p in bbox]
        
        # Draw filled polygon
        overlay_draw.polygon(polygon, fill=white_with_opacity)
    
    # Composite the overlay onto the image
    image = Image.alpha_composite(image, overlay)
    
    # Convert back to RGB for saving as JPG if needed
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        # For JPG, we need to composite on a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        background.save(output_path, quality=95)
    else:
        image.save(output_path)


def main():
    """Main entry point for the manga translation tool."""
    parser = argparse.ArgumentParser(
        description="Detect Japanese text in manga images, extract via OCR, and white out text areas"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the input manga image file"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: File not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: Path is not a file: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Processing: {input_path}", file=sys.stderr)
        
        # Detect text regions
        text_regions = detect_text_regions(input_path)
        
        if not text_regions:
            print("No text detected in the image.", file=sys.stderr)
            sys.exit(1)
        
        # Sort regions in manga reading order
        sorted_regions = sort_text_regions(text_regions)
        
        # Extract text using MangaOCR (handles vertical text)
        extracted_texts = extract_text(input_path, sorted_regions)
        
        # Get translations with full context
        translations = translate_with_context(extracted_texts)
        
        print("=== Detected Japanese Text (in reading order) ===\n")
        for i, (text, region, translation) in enumerate(zip(extracted_texts, sorted_regions, translations), 1):
            bbox = region[0]
            orientation = "vertical" if is_vertical_text(bbox) else "horizontal"
            hiragana = convert_to_hiragana(text)
            romaji = convert_to_romaji(text)
            
            print(f"{i}. [{orientation}]")
            print(f"   Japanese: {text}")
            print(f"   Hiragana: {hiragana}")
            print(f"   Romaji:   {romaji}")
            print(f"   English:  {translation}")
            print()
        
        # Create output image with whited-out text
        extension = input_path.suffix.lower()
        if not extension:
            extension = ".png"
        
        output_filename = f"output{extension}"
        output_path = input_path.parent / output_filename
        
        whiteout_text_areas(input_path, output_path, sorted_regions, opacity=0.8)
        
        print(f"\n✓ Processed image saved to: {output_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
