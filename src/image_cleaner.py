"""Image cleaning module for removing and replacing text areas."""

import sys
import numpy as np
from PIL import Image, ImageDraw


def sample_background_color(image_array, bbox, border_px=5):
    """Sample the dominant background color from a border inside the text region.
    
    Samples from a border area (5px inset) inside the text box to get representative 
    background without sampling the main text in the center.
    Ignores dark/black pixels with low weight to avoid sampling text.
    
    Args:
        image_array: numpy array of the image
        bbox: bounding box coordinates
        border_px: pixels to inset from edge (samples from this border)
    
    Returns:
        tuple of (R, G, B) color values
    """
    try:
        points = np.array(bbox, dtype=np.int32)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        h, w = image_array.shape[:2]
        
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Inset by border_px to create a border area inside the text box
        inset_x_min = min(x_min + border_px, x_max - 1)
        inset_y_min = min(y_min + border_px, y_max - 1)
        inset_x_max = max(x_max - border_px, x_min + 1)
        inset_y_max = max(y_max - border_px, y_min + 1)
        
        # Sample from the border areas
        samples = []
        
        # Top border
        if inset_y_min > y_min:
            samples.append(image_array[y_min:inset_y_min, x_min:x_max])
        # Bottom border
        if inset_y_max < y_max:
            samples.append(image_array[inset_y_max:y_max, x_min:x_max])
        # Left border
        if inset_x_min > x_min:
            samples.append(image_array[y_min:y_max, x_min:inset_x_min])
        # Right border
        if inset_x_max < x_max:
            samples.append(image_array[y_min:y_max, inset_x_max:x_max])
        
        if not samples:
            return (255, 255, 255)
        
        # Concatenate all border samples
        sampled = np.concatenate([s.reshape(-1, s.shape[-1]) for s in samples])
        
        # Calculate brightness for each pixel (average of RGB)
        brightness = np.mean(sampled[:, :3], axis=1)
        
        # Create weight array: dark pixels get 0.2 weight, others get 1.0
        # Dark is defined as brightness < 50 (out of 255)
        weights = np.where(brightness < 50, 0.2, 1.0)
        
        # Apply weighted average
        weighted_samples = sampled[:, :3] * weights[:, np.newaxis]
        weighted_sum = np.sum(weighted_samples, axis=0)
        weight_total = np.sum(weights)
        
        if weight_total > 0:
            avg_color = (weighted_sum / weight_total).astype(int)
        else:
            avg_color = np.mean(sampled[:, :3], axis=0).astype(int)
        
        return tuple(avg_color)
    
    except Exception as e:
        print(f"Error sampling background color: {e}", file=sys.stderr)
        return (255, 255, 255)


def whiteout_text_areas(image_path, output_path, text_regions, opacity=0.8):
    """Clean text areas with background color and rounded corners.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        text_regions: List of detected text regions with bboxes
        opacity: Opacity of overlay (0.0-1.0), where 1.0 is fully opaque
    """
    # Open image and convert to RGBA for transparency support
    image = Image.open(image_path).convert('RGBA')
    img_array = np.array(Image.open(image_path))
    width, height = image.size
    
    # Create overlay layer
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Calculate alpha value (0-255 where 255 is fully opaque)
    alpha = int(255 * opacity)
    
    # Process each text region
    for region in text_regions:
        bbox = region[0]
        
        # Convert bbox points to polygon coordinates
        polygon = [(int(p[0]), int(p[1])) for p in bbox]
        
        # Sample background color from around the region
        bg_color = sample_background_color(img_array, bbox)
        overlay_color = bg_color + (alpha,)  # Add alpha channel
        
        # Draw filled polygon
        overlay_draw.polygon(polygon, fill=overlay_color)
    
    # Composite the overlay onto the image
    image = Image.alpha_composite(image, overlay)
    
    # Convert back to RGB for saving as JPG if needed
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        # For JPG, composite on a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        background.save(output_path, quality=95)
    else:
        image.save(output_path)
