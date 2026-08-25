"""Image cleaning module for removing and replacing text areas."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from src.region_data import RegionData
from src.inpainting import inpaint_regions


def sample_background_color(image_array, polygon, border_px=5):
    """Sample a representative background color for a (possibly non-rectangular) polygon.

    This simplified approach samples pixels inside the polygon but near its edges by
    taking the top/bottom/left/right border bands of the polygon's bounding box and
    keeping only pixels that fall inside the polygon mask. Dark pixels are down-weighted
    to avoid sampling text strokes.

    Args:
        image_array: numpy array of the image (H, W, C)
        polygon: sequence of (x, y) points describing the polygon
        border_px: thickness in pixels of border bands to sample from

    Returns:
        tuple of (R, G, B) color values
    """
    try:
        pts = np.array(polygon, dtype=np.int32)
        if pts.size == 0:
            return (255, 255, 255)

        x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
        y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())

        h, w = image_array.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Build polygon mask (full image size) and then slice the bounding box
        mask_img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask_img).polygon([tuple(p) for p in pts], fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)

        samples = []

        # Top band
        top_y = min(y_min + border_px, y_max)
        if top_y > y_min:
            region = image_array[y_min:top_y, x_min:x_max]
            region_mask = mask[y_min:top_y, x_min:x_max]
            if region_mask.any():
                samples.append(region[region_mask])

        # Bottom band
        bot_y = max(y_max - border_px, y_min)
        if bot_y < y_max:
            region = image_array[bot_y:y_max, x_min:x_max]
            region_mask = mask[bot_y:y_max, x_min:x_max]
            if region_mask.any():
                samples.append(region[region_mask])

        # Left band
        left_x = min(x_min + border_px, x_max)
        if left_x > x_min:
            region = image_array[y_min:y_max, x_min:left_x]
            region_mask = mask[y_min:y_max, x_min:left_x]
            if region_mask.any():
                samples.append(region[region_mask])

        # Right band
        right_x = max(x_max - border_px, x_min)
        if right_x < x_max:
            region = image_array[y_min:y_max, right_x:x_max]
            region_mask = mask[y_min:y_max, right_x:x_max]
            if region_mask.any():
                samples.append(region[region_mask])

        if not samples:
            return (255, 255, 255)

        sampled = np.concatenate([s.reshape(-1, s.shape[-1]) for s in samples])

        brightness = np.mean(sampled[:, :3], axis=1)
        weights = np.where(brightness < 50, 0.2, 1.0)

        weighted_samples = sampled[:, :3] * weights[:, np.newaxis]
        weighted_sum = np.sum(weighted_samples, axis=0)
        weight_total = np.sum(weights)

        if weight_total > 0:
            avg_color = (weighted_sum / weight_total).astype(int)
        else:
            avg_color = np.mean(sampled[:, :3], axis=0).astype(int)

        if all(int(c) > 230 for c in avg_color):
            return (255, 255, 255)

        return tuple(int(c) for c in avg_color)

    except Exception as e:
        print(f"Error sampling background color: {e}", file=sys.stderr)
        return (255, 255, 255)


def extract_bbox(region):
    if isinstance(region, np.ndarray):
        if region.ndim == 2 and region.shape[1] == 2:
            return [(int(point[0]), int(point[1])) for point in region]
        region = region.tolist()

    if isinstance(region, dict):
        region = region.get("bbox", region)

    if isinstance(region, (list, tuple)):
        if len(region) == 0:
            return []

        # Region may be a direct list of points
        if all(
            isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2
            for point in region
        ):
            return [(int(point[0]), int(point[1])) for point in region]

        # Region may be a tuple like (bbox, text, confidence)
        first = region[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, (list, tuple)) and len(first) > 0 and all(
            isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2
            for point in first
        ):
            return [(int(point[0]), int(point[1])) for point in first]

    raise ValueError(f"Unable to extract bbox from region: {region}")


def region_bg_color(region: RegionData, image_array: np.ndarray) -> tuple:
    region_color_override = region.clean_color_override

    # Allow a per-region free-form selection to override the bbox.
    selection = getattr(region, "clean_shape_selection", None)
    if selection:
        polygon = [(int(p[0]), int(p[1])) for p in selection]
        sample_target = polygon
    else:
        bbox = region.bbox
        polygon = [(int(p[0]), int(p[1])) for p in bbox]
        sample_target = bbox

    if region_color_override is None:
        bg_color = sample_background_color(image_array, sample_target)
    else:
        bg_color = region_color_override

    return bg_color


def clean_image(image, text_regions, opacity=0.8):
    """Return a PIL Image with text regions cleaned using background colour.

    Args:
        image: PIL Image object
        text_regions: list of bounding polygons, region tuples, or RegionData objects
        opacity: default overlay opacity between 0.0 and 1.0
    """
    image = image.convert('RGBA')
    rgb_image = image.convert('RGB')
    img_array = np.array(rgb_image)

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    redraw_polygons = []

    for region in text_regions:
        if region.clean_region is False and region.redraw_region is False:
            continue
        region_color_override = region.clean_color_override

        # Allow a per-region free-form selection to override the bbox.
        selection = getattr(region, "clean_shape_selection", None)
        if selection:
            polygon = [(int(p[0]), int(p[1])) for p in selection]
            sample_target = polygon
        else:
            bbox = region.bbox
            polygon = [(int(p[0]), int(p[1])) for p in bbox]
            sample_target = bbox

        if region.redraw_region:
            redraw_polygons.append(polygon)
            continue

        if region_color_override is None:
            bg_color = sample_background_color(img_array, sample_target)
        else:
            bg_color = region_color_override
        alpha = int(255 * region.clean_opacity)
        overlay_color = bg_color + (alpha,)
        overlay_draw.polygon(polygon, fill=overlay_color)
    
    # Try to inpaint the region using a local inpainting model.
    if len(redraw_polygons):
        try:
            res = inpaint_regions(rgb_image, redraw_polygons)
            if res is not None:
                final_crop, (x_off, y_off) = res
                # Paste result directly onto the image (preserve alpha)
                final_rgba = final_crop.convert("RGBA")
                image.paste(final_rgba, (x_off, y_off), final_rgba)
        except Exception as e:
            print(f"Error during inpainting: {e}", file=sys.stderr)

    return Image.alpha_composite(image, overlay)
