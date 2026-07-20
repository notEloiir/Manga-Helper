#!/usr/bin/env python3
"""CLI tool for manga translation: detect Japanese text, extract via OCR, and white out text areas."""

import sys
import argparse
from pathlib import Path

# Import from modules
from src.ocr import detect_text_regions, sort_text_regions, extract_text, is_vertical_text
from src.text_processor import convert_to_hiragana, convert_to_romaji, translate_with_context
from src.image_cleaner import whiteout_text_areas


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
        
        # Create output image with cleaned text
        extension = input_path.suffix.lower()
        if not extension:
            extension = ".png"
        
        output_filename = f"output{extension}"
        output_path = input_path.parent / output_filename
        
        whiteout_text_areas(input_path, output_path, sorted_regions, opacity=0.9)
        
        print(f"\n✓ Processed image saved to: {output_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
