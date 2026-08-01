# Manga Helper

A CLI tool for learning and assisting with manga translation. It extracts Japanese text from manga images and provides multiple representations to help readers who understand spoken Japanese but need help with kanji.

## Features

- **Text Detection**: Finds all text regions in manga images using EasyOCR
- **OCR Extraction**: Recognizes Japanese text (both vertical and horizontal) using MangaOCR
- **Learning Aids**: 
  - Shows original kanji text
  - Converts kanji to hiragana (phonetic representation) for pronunciation help
  - Romanizes text to romaji (Latin characters) for reference
  - Provides English translation for context
- **Image Cleanup**: Removes text from images by:
  - Sampling the background color from each text region
  - Replacing text with the appropriate background color
  - Using semi-transparent overlays (80% opacity) for natural blending
- **Manga Reading Order**: Sorts detected text regions in proper manga reading order (top-to-bottom, right-to-left)

## Installation

Using `uv` package manager:

```bash
uv pip install -e .
```

Or with standard pip:

```bash
pip install -e .
```

## Usage

```bash
manga-translator <image_path>
```

### Example

```bash
manga-translator sample_input.jpg
```

### Output

The tool produces:

1. **Console output** with detected text for each region:
   ```
   === Detected Japanese Text (in reading order) ===
   
   1. [horizontal]
      Japanese: こんにちは
      Hiragana: こんにちは
      Romaji:   konnichiha
      English:  Hello
   ```

2. **Output image** (`output.jpg`) with text areas cleaned and blended with the background

## Project Structure

- **[main.py](main.py)** - CLI entry point and workflow orchestration
- **[ocr.py](ocr.py)** - Text detection and OCR extraction
- **[text_processor.py](text_processor.py)** - Japanese text conversion and translation
- **[image_cleaner.py](image_cleaner.py)** - Image manipulation and text area cleanup

## Dependencies

- **pillow** - Image processing
- **easyocr** - Text region detection
- **manga-ocr** - Japanese text recognition (handles vertical text)
- **numpy** - Numerical operations
- **kanjiconv** - Kanji to hiragana/romaji conversion
- **deep-translator** - English translation

## Technical Details

### Text Detection

Uses EasyOCR with Japanese language support and tuned sensitivity parameters:
- `contrast_ths=0.05` - Detect lower contrast text
- `text_threshold=0.5` - Lower confidence threshold
- `low_text=0.3` - Enhanced sensitivity

### Text Extraction

MangaOCR is used for actual character recognition because it:
- Handles both horizontal and vertical Japanese text
- Is specialized for manga with various fonts and styles
- Supports vertical text without manual rotation

### Background Color Sampling

For each text region:
1. Samples a 5px border inside the text box
2. Weights dark pixels (< 50 brightness) at 0.2x to avoid sampling text
3. Uses weighted average to get the representative background color

### Image Cleanup

- Overlays semi-transparent color matching the background
- Maintains 80% opacity for natural blending
- Outputs as JPEG with 95% quality or PNG depending on input

## Requirements

- Python 3.10+
- ~2GB disk space for OCR models (downloaded on first run)
- GPU support optional (uses CPU by default)

## Use Cases

- **Learning kanji**: See hiragana and romaji alongside kanji to understand pronunciation
- **Reading comprehension**: Get English translations for unfamiliar words while reading
- **Text extraction**: Extract raw Japanese text from manga for studying
- **Clean scans**: Remove text from manga panels for re-translation or studying art

## Notes

- First run will download OCR models (~1-2GB), which may take several minutes
- Vertical text detection uses a 1.2x height-to-width ratio threshold
- English translation uses Google Translate via deep-translator (intended for context, not professional translation)
- Works best with manga that has clear, distinct text areas
- Hiragana conversion helps readers who know spoken Japanese but struggle with kanji

## Fonts

The GUI includes built-in support for fonts from your system and from the `./fonts/` directory. Fonts suitable for manga/comic text styling are automatically discovered from system paths. Any fonts you add to the `./fonts/` directory will be available in the font selector without filtering.

### Included Font Licenses

The default font - CC Wild Words Roman, included in the `./fonts/` directory - is licensed under the **SIL Open Font License (OFL)**, which permits free personal and commercial use. See `fonts/OFL.txt` for full license details.

## License

MIT
