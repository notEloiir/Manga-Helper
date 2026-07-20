"""Text processing module for Japanese text conversion and translation."""

import sys
from kanjiconv import KanjiConv
from deep_translator import GoogleTranslator


# Global converter (initialized once for performance)
_kanjiconv = None


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
