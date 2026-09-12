#!/usr/bin/env python3
"""CLI tool for manga translation: detect Japanese text, extract via OCR, and white out text areas."""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Import from modules
from src.ocr import detect_text_regions, sort_text_regions, extract_text, is_vertical_text
from src.text_processor import convert_to_hiragana, convert_to_romaji, translate
from src.image_cleaner import clean_image, extract_bbox
from src.region_data import RegionData


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

        image = Image.open(input_path)
        
        # Detect text regions
        image_array = np.array(image)
        text_regions = detect_text_regions(image_array)
        
        if not text_regions:
            print("No text detected in the image.", file=sys.stderr)
            sys.exit(1)
        
        # Sort regions in manga reading order
        sorted_regions = sort_text_regions(text_regions)
        
        # Extract text using MangaOCR (handles vertical text)
        extracted_texts = extract_text(image, sorted_regions)
        region_data_list = [
            RegionData(
                normalized_bbox=[
                    (min(point[0] for bbox in detected_bboxes for point in bbox), min(point[1] for bbox in detected_bboxes for point in bbox)),
                    (max(point[0] for bbox in detected_bboxes for point in bbox), min(point[1] for bbox in detected_bboxes for point in bbox)),
                    (max(point[0] for bbox in detected_bboxes for point in bbox), max(point[1] for bbox in detected_bboxes for point in bbox)),
                    (min(point[0] for bbox in detected_bboxes for point in bbox), max(point[1] for bbox in detected_bboxes for point in bbox)),
                ],
                detected_bboxes=detected_bboxes,
                ocr_text=text,
                clean_region=True,
                clean_opacity=0.9
                )
            for region, text in zip(sorted_regions, extracted_texts)
            for detected_bboxes in [[[(int(point[0]), int(point[1])) for point in bbox] for bbox in region[0]]]
        ]
        
        # Get translations with full context
        translations = translate(extracted_texts)
        
        print("=== Detected Japanese Text (in reading order) ===\n")
        for i, (text, region, translation) in enumerate(zip(extracted_texts, sorted_regions, translations), 1):
            bbox = [point for detected_bbox in region[0] for point in detected_bbox]
            orientation = "vertical" if is_vertical_text(bbox) else "horizontal"
            hiragana = convert_to_hiragana(text)
            romaji = convert_to_romaji(text)

            region_data_list[i-1].hiragana = hiragana
            region_data_list[i-1].romaji = romaji
            
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
        
        cleaned = clean_image(image, region_data_list, opacity=0.8)

        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            background = Image.new('RGB', cleaned.size, (255, 255, 255))
            background.paste(cleaned, mask=cleaned.split()[3])
            background.save(output_path, quality=95)
        else:
            cleaned.save(output_path)
        
        print(f"\nProcessed image saved to: {output_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
