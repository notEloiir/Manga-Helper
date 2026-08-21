"""Local manga inpainting support using a cached LaMa model.

This module keeps the existing project API (`inpaint_region`) but swaps the
missing diffusers SD pipeline for a local, cached LaMa inpainting backend.
The model is stored under the same project-local cache pattern used elsewhere
in the app and is only fetched once if it is not already present.
"""
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INPAINT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "inpaint"
_LAMA_MODEL_PATH = _INPAINT_CACHE_DIR / "lama" / "big-lama.pt"
_LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)

_INPAINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LAMA_MODEL", str(_LAMA_MODEL_PATH))

_inpaint_pipe = None


def _get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_lama_model():
    """Ensure a local LaMa checkpoint exists in the project cache."""
    _LAMA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LAMA_MODEL_PATH.exists():
        os.environ["LAMA_MODEL"] = str(_LAMA_MODEL_PATH)
        return str(_LAMA_MODEL_PATH)

    print(f"Downloading LaMa model to: {_LAMA_MODEL_PATH}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(_LAMA_MODEL_URL, str(_LAMA_MODEL_PATH))
        os.environ["LAMA_MODEL"] = str(_LAMA_MODEL_PATH)
        return str(_LAMA_MODEL_PATH)
    except Exception as e:
        print(f"Failed to download LaMa model: {e}", file=sys.stderr)
        return None


def get_inpaint_pipeline(model_name=None):
    """Return a cached local LaMa model pipeline if available."""
    global _inpaint_pipe
    if _inpaint_pipe is not None:
        return _inpaint_pipe

    try:
        from lama_cleaner.model_manager import ModelManager
    except Exception as e:
        print(f"Inpainting unavailable (lama_cleaner missing): {e}", file=sys.stderr)
        return None

    local_model = _ensure_lama_model()
    if local_model is None:
        return None

    try:
        _inpaint_pipe = ModelManager("lama", _get_device_name())
        return _inpaint_pipe
    except Exception as e:
        print(f"Failed to initialize LaMa pipeline: {e}", file=sys.stderr)
        return None


def _make_square_canvas(img, mask):
    """Create a square canvas with image and mask pasted top-left and return both and the square size."""
    w, h = img.size
    size = max(w, h)
    canvas_img = Image.new("RGB", (size, size), (255, 255, 255))
    canvas_mask = Image.new("L", (size, size), 0)
    canvas_img.paste(img, (0, 0))
    canvas_mask.paste(mask, (0, 0))
    return canvas_img, canvas_mask, size


def inpaint_regions(image: Image.Image, polygons, prompt="clean manga background", output_size=512):
    """Attempt to inpaint one or more polygon regions in `image`.

    Accepts either a single polygon or a list of polygons. Returns a tuple
    `(final_crop, (x_min, y_min))` on success, or `None` on failure.
    """
    pipe = get_inpaint_pipeline()
    if pipe is None:
        return None

    all_pts = np.concatenate([np.asarray(poly, dtype=np.int32) for poly in polygons])
    x_min, y_min = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    x_max, y_max = int(all_pts[:, 0].max()), int(all_pts[:, 1].max())

    padding = 8
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.width, x_max + padding)
    y_max = min(image.height, y_max + padding)

    cropped = image.crop((x_min, y_min, x_max, y_max)).convert("RGB")

    # Create mask: white (255) inside polygon(s), black outside
    mask = Image.new("L", cropped.size, 0)
    for poly in polygons:
        shifted = [(int(p[0]) - x_min, int(p[1]) - y_min) for p in poly]
        ImageDraw.Draw(mask).polygon(shifted, fill=255)

    # Make square canvas and resize to model-friendly size
    canvas_img, canvas_mask, size = _make_square_canvas(cropped, mask)

    try:
        from lama_cleaner.schema import Config, HDStrategy

        resized_img = canvas_img.resize((output_size, output_size), resample=Image.LANCZOS)
        resized_mask = canvas_mask.resize((output_size, output_size), resample=Image.NEAREST)

        image_np = np.array(resized_img)
        mask_np = np.array(resized_mask)

        config = Config(
            ldm_steps=20,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )

        result = pipe(image_np, mask_np, config)
        out_img = Image.fromarray(result[:, :, ::-1])

        out_resized = out_img.resize((size, size), resample=Image.LANCZOS)
        final_crop = out_resized.crop((0, 0, cropped.size[0], cropped.size[1]))
        return final_crop, (x_min, y_min)

    except Exception as e:
        print(f"Inpainting failed: {e}", file=sys.stderr)
        return None
