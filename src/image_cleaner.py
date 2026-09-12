"""Image cleaning module for removing and replacing text areas."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageChops

from src.region_data import RegionData
from src.inpainting import inpaint_regions


# Booyer-Moore
def majority_element(arr):
    candidate = None
    count = 0

    for x in arr:
        if count == 0:
            candidate = x
            count = 1
        elif (x == candidate).all():
            count += 1
        else:
            count -= 1

    return candidate


def _sample_background_pixels(image_array, polygon, border_px=5):
    """Collect background candidate pixels from a polygon's edge bands."""
    pts = np.array(polygon, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, image_array.shape[-1]), dtype=image_array.dtype)

    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())

    h, w = image_array.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    mask_img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask_img).polygon([tuple(p) for p in pts], fill=1)
    mask = np.array(mask_img, dtype=np.uint8).astype(bool)

    samples = []

    top_y = min(y_min + border_px, y_max)
    if top_y > y_min:
        region = image_array[y_min:top_y, x_min:x_max]
        region_mask = mask[y_min:top_y, x_min:x_max]
        if region_mask.any():
            samples.append(region[region_mask])

    bot_y = max(y_max - border_px, y_min)
    if bot_y < y_max:
        region = image_array[bot_y:y_max, x_min:x_max]
        region_mask = mask[bot_y:y_max, x_min:x_max]
        if region_mask.any():
            samples.append(region[region_mask])

    left_x = min(x_min + border_px, x_max)
    if left_x > x_min:
        region = image_array[y_min:y_max, x_min:left_x]
        region_mask = mask[y_min:y_max, x_min:left_x]
        if region_mask.any():
            samples.append(region[region_mask])

    right_x = max(x_max - border_px, x_min)
    if right_x < x_max:
        region = image_array[y_min:y_max, right_x:x_max]
        region_mask = mask[y_min:y_max, right_x:x_max]
        if region_mask.any():
            samples.append(region[region_mask])

    if not samples:
        return np.empty((0, image_array.shape[-1]), dtype=image_array.dtype)

    return np.concatenate([sample.reshape(-1, sample.shape[-1]) for sample in samples])


def sample_background_color(image_array, polygon, border_px=5):
    """Sample a representative background color for one or more polygons.

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
        if isinstance(polygon, np.ndarray):
            polygon = polygon.tolist()
        if not polygon:
            return (255, 255, 255)

        first_point = polygon[0]
        is_polygon_list = (
            isinstance(first_point, (list, tuple, np.ndarray))
            and len(first_point) > 0
            and isinstance(first_point[0], (list, tuple, np.ndarray))
        )
        polygons = polygon if is_polygon_list else [polygon]
        samples = [_sample_background_pixels(image_array, item, border_px) for item in polygons]
        samples = [sample for sample in samples if sample.size]
        if not samples:
            return (255, 255, 255)

        sampled = np.concatenate(samples)
        sample_size = sampled.shape[0]

        majority_color = majority_element(sampled[:, :3])
        majority_size = np.count_nonzero(sampled[:, :3] == majority_color)
        if majority_color is not None and majority_size >= sample_size / 2:
            return tuple(int(c) for c in majority_color)

        avg_color = np.mean(sampled[:, :3], axis=0).astype(int)

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

    polygons = cleaning_polygons(region)
    sample_target = polygons[0]

    if region_color_override is None:
        bg_color = sample_background_color(image_array, sample_target)
    else:
        bg_color = region_color_override

    return bg_color


def cleaning_polygons(region: RegionData) -> list:
    """Return the polygons used to clean a region, in priority order."""
    selection = region.clean_shape_selection
    if selection is not None:
        return [selection]

    detected_bboxes = region.detected_bboxes or []
    if detected_bboxes:
        return detected_bboxes

    bbox = region.normalized_bbox
    return [bbox] if bbox else []


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
        polygons = cleaning_polygons(region)
        if not polygons:
            continue

        if region_color_override is None:
            bg_color = sample_background_color(img_array, polygons)
        else:
            bg_color = region_color_override

        for polygon in polygons:
            if region.redraw_region:
                redraw_polygons.append(polygon)
                continue

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
                # But only for redrawn polygons to not degrade image quality
                patch = final_crop.convert("RGBA")
                poly_mask = Image.new("L", patch.size, 0)
                draw_mask = ImageDraw.Draw(poly_mask)

                for poly in redraw_polygons:
                    local_poly = [
                        (x - x_off, y - y_off)
                        for x, y in poly
                    ]
                    draw_mask.polygon(local_poly, fill=255)

                # Respect both polygon selection and patch alpha
                alpha = patch.getchannel("A")
                final_mask = ImageChops.multiply(alpha, poly_mask)

                image.paste(patch, (x_off, y_off), final_mask)

        except Exception as e:
            print(f"Error during inpainting: {e}", file=sys.stderr)

    return Image.alpha_composite(image, overlay)
