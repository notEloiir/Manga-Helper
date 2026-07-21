"""Text processing module for Japanese text conversion and offline translation."""

import os
import sys
from pathlib import Path

import torch
from kanjiconv import KanjiConv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Global converter and translator resources (initialized once for performance)
_kanjiconv = None
_translation_model = None
_translation_tokenizer = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRANSFORMERS_CACHE_DIR = _PROJECT_ROOT / ".cache" / "transformers"
_TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TRANSFORMERS_CACHE", str(_TRANSFORMERS_CACHE_DIR))


def _get_device_index():
    if torch.cuda.is_available():
        return 0
    return -1


def _get_translation_resources():
    global _translation_model, _translation_tokenizer
    if _translation_model is None or _translation_tokenizer is None:
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        _translation_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(_TRANSFORMERS_CACHE_DIR),
        )
        _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(_TRANSFORMERS_CACHE_DIR),
        )
    return _translation_model, _translation_tokenizer


def convert_to_hiragana(text):
    """Convert Japanese text (kanji + hiragana) to hiragana phonetics.
    
    Uses kanjiconv to convert all characters to hiragana.
    """
    global _kanjiconv
    if _kanjiconv is None:
        _kanjiconv = KanjiConv(separator="")
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
        _kanjiconv = KanjiConv(separator=" ")
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
        model, tokenizer = _get_translation_resources()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        translations = []
        for text in texts:
            if not text.strip():
                translations.append("")
                continue
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                inputs = inputs.to(device)
                model.to(device)
                outputs = model.generate(**inputs)
                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                translations.append(translated)
            except Exception as e:
                print(f"Error translating text: {e}", file=sys.stderr)
                translations.append("")
        return translations
    except Exception as e:
        print(f"Error initializing translator: {e}", file=sys.stderr)
        return [""] * len(texts)
