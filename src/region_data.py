from dataclasses import dataclass


@dataclass
class RegionData:
    bbox: list
    ocr_text: str = ""
    hiragana: str = ""
    romaji: str = ""
    translation: str = ""
    user_text: str = ""
    orientation: str = "horizontal"
    confidence: float | None = None
    to_update: bool = False
    clean_region: bool = True
    redraw_region: bool = False
    clean_opacity: float = 0.9
    clean_color_override: tuple | None = None
    clean_shape_selection: list | None = None
    # Per-region text settings
    font_family: str = ""
    font_size: int = 24
    text_angle: float = 0.0
    text_center: bool = True
    text_align_center: bool = True
    text_color: tuple = (0, 0, 0)
    text_bg_color: tuple = (255, 255, 255)
