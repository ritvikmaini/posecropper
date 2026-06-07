"""Classical-CV product-segmentation cascade (Otsu -> K-Means -> default), with GrabCut
implemented but disabled in favor of a lighter fallback. EXPLORED AND NOT IN THE LIVE
PATH: P-codes route to the default crop (see router.py). Kept to show the approach and
its luminance-adaptive diagnostic visualization. Deps: numpy + opencv only."""
import os

import cv2
import numpy as np

from posecropper.taxonomy import ASPECT_RATIO, MAX_AREA_RATIO, MIN_AREA_RATIO, PRODUCT_PADDING


def get_largest_product_box(image_rgb, image_height, image_width):
    """
    Attempts to get the largest bounding box using Otsu's binarization,
    GrabCut segmentation, K-Means segmentation, and if all fail,
    applies default cropping.
    """
    print("Starting get_largest_product_box")

    # Convert RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Try Otsu's Binarization first
    bounding_box = otsu_binarization(image_bgr)
    method_used = "Otsu Binarization"

    if bounding_box is not None and is_valid_bounding_box(bounding_box, image_height, image_width):
        print(f"{method_used} succeeded. Bounding box: {bounding_box}")
        return bounding_box, method_used

    # Try K-Means Segmentation if Otsu fails
    bounding_box = kmeans_segmentation(image_bgr)
    method_used = "K-Means Segmentation"

    if bounding_box is not None and is_valid_bounding_box(bounding_box, image_height, image_width):
        print(f"{method_used} succeeded. Bounding box: {bounding_box}")
        return bounding_box, method_used

    #TODO: find a more lightweight fallback method
    """
    #Try GrabCut Segmentation if Otsu fails
    bounding_box = grabcut_segmentation(image_bgr)
    method_used = "GrabCut Segmentation"

    if bounding_box is not None and is_valid_bounding_box(bounding_box, image_height, image_width):
        print(f"{method_used} succeeded. Bounding box: {bounding_box}")
        return bounding_box, method_used
    """
    # Final fallback - Default Cropping
    method_used = "Default Cropping"
    x1, y1 = 0, 0
    x2 = image_width
    y2 = image_height
    bounding_box = [x1, y1, x2, y2]
    print(f"{method_used} used. Bounding box: {bounding_box}")
    return bounding_box, method_used


def is_valid_bounding_box(bounding_box, image_height, image_width, min_ratio=MIN_AREA_RATIO, max_ratio=MAX_AREA_RATIO):
    x1, y1, x2, y2 = bounding_box
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_height * image_width
    area_ratio = box_area / image_area
    return float(min_ratio) <= float(area_ratio) <= float(max_ratio)


def otsu_binarization(image_bgr):
    print("Attempting Otsu's Binarization without Histogram Equalization")
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Otsu's Binarization failed: No contours found.")
        return None
    # Find and return the largest contour's bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return [x, y, x + w, y + h]


def grabcut_segmentation(image_bgr):
    print("Attempting GrabCut Segmentation")
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    rect = (1, 1, image_bgr.shape[1] - 2, image_bgr.shape[0] - 2)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        segmented = image_bgr * mask_fg[:, :, np.newaxis]
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("GrabCut segmentation failed: No contours found.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, x + w, y + h]
    except Exception as e:
        print(f"GrabCut segmentation failed: {e}")
        return None


def kmeans_segmentation(image_bgr, k=2):
    print("Attempting K-Means Segmentation with k=2")

    # Prepare image for K-means
    Z = image_bgr.reshape((-1, 3)).astype(np.float32)

    try:
        _, labels, centers = cv2.kmeans(
            Z, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10, cv2.KMEANS_RANDOM_CENTERS
        )
    except Exception as e:
        print(f"K-Means clustering failed: {e}")
        return None

    labels = labels.flatten()
    valid_bounding_boxes = []
    for i in range(k):
        mask = (labels == i).reshape(image_bgr.shape[:2]).astype('uint8') * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_box = [x, y, x + w, y + h]
            if is_valid_bounding_box(bounding_box, image_bgr.shape[0], image_bgr.shape[1]):
                valid_bounding_boxes.append(bounding_box)

    if valid_bounding_boxes:
        largest_box = max(valid_bounding_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return largest_box
    else:
        print("No valid bounding boxes found in K-Means segmentation.")
        return None


def crop_product(image_rgb, image_height, image_width, bounding_box, default_method=True, target_aspect_ratio=ASPECT_RATIO, padding_ratio=PRODUCT_PADDING):
    print("Starting crop_product")
    print(f"Received bounding box: {bounding_box}")

    if not bounding_box or len(bounding_box) != 4:
        raise ValueError(f"Invalid bounding box format: {bounding_box}")

    x1, y1, x2, y2 = bounding_box
    box_width = x2 - x1
    box_height = y2 - y1

    center_x, center_y = x1 + box_width // 2, y1 + box_height // 2
    print(f"Center of bounding box: ({center_x}, {center_y})")

    if not default_method:
        # Apply padding
        padded_width = int(box_width * (1 + 2 * padding_ratio))
        padded_height = int(box_height * (1 + 2 * padding_ratio))
    else:
        # No padding, use box dimensions
        padded_width = box_width
        padded_height = box_height

    # Adjust dimensions to maintain aspect ratio and fit within image bounds
    # Calculate the maximum possible dimensions
    max_width = min(padded_width, image_width)
    max_height = min(padded_height, image_height)

    # Calculate adjusted dimensions based on aspect ratio
    if (max_height / max_width) >= target_aspect_ratio:
        # Height is the limiting factor
        adjusted_height = max_height
        adjusted_width = int(adjusted_height / target_aspect_ratio)
        if adjusted_width > image_width:
            adjusted_width = image_width
            adjusted_height = int(adjusted_width * target_aspect_ratio)
    else:
        # Width is the limiting factor
        adjusted_width = max_width
        adjusted_height = int(adjusted_width * target_aspect_ratio)
        if adjusted_height > image_height:
            adjusted_height = image_height
            adjusted_width = int(adjusted_height / target_aspect_ratio)

    # Center the crop
    x1_new = center_x - adjusted_width // 2
    x2_new = center_x + adjusted_width // 2
    y1_new = center_y - adjusted_height // 2
    y2_new = center_y + adjusted_height // 2

    # Ensure within image bounds
    x1_new = max(0, int(x1_new))
    y1_new = max(0, int(y1_new))
    x2_new = min(image_width, int(x2_new))
    y2_new = min(image_height, int(y2_new))

    # Recalculate adjusted dimensions
    adjusted_width = x2_new - x1_new
    adjusted_height = y2_new - y1_new

    # Recalculate aspect ratio
    actual_aspect_ratio = adjusted_height / adjusted_width
    crop_area_percentage = (adjusted_width * adjusted_height) / (image_width * image_height) * 100

    print(f"Final crop coordinates: ({x1_new}, {y1_new}, {x2_new}, {y2_new})")
    print(f"Adjusted aspect ratio: {actual_aspect_ratio:.2f}, Expected aspect ratio: {target_aspect_ratio}")
    print(f"Crop area percentage: {crop_area_percentage:.2f}%")

    cropped_image = image_rgb[y1_new:y2_new, x1_new:x2_new]
    crop_details = {
        "initial_bounding_box": bounding_box,
        "adjusted_bounding_box": [x1_new, y1_new, x2_new, y2_new],
        "adjusted_box_width": adjusted_width,
        "adjusted_box_height": adjusted_height,
        "actual_aspect_ratio": actual_aspect_ratio,
        "expected_aspect_ratio": target_aspect_ratio,
        "padding": padding_ratio if not default_method else 0,
        "crop_area_percentage": crop_area_percentage
    }

    print("crop_product completed successfully")
    return cropped_image, crop_details


def visualize_product_segmentation(image_source, image_path):
    """Generate a diagnostic visualization of the product segmentation on the image at image_path.
    image_source: an ImageSource instance (posecropper.io.base.ImageSource) — not a GCS bucket."""
    image_rgb = image_source.read(image_path)
    if image_rgb is None:
        print(f"Could not load image from path: {image_path}")
        return None

    image_height, image_width, _ = image_rgb.shape
    bounding_box, method_used = get_largest_product_box(image_rgb, image_height, image_width)
    if bounding_box is None:
        print("No valid bounding box detected.")
        return None

    default_method = method_used == "Default Cropping"
    cropped_image, crop_details = crop_product(image_rgb, image_height, image_width, bounding_box, default_method)
    adjusted_bounding_box = crop_details['adjusted_bounding_box']
    crop_area_percentage = crop_details['crop_area_percentage']

    visualization = image_rgb.copy()
    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)

    adj_x1, adj_y1, adj_x2, adj_y2 = adjusted_bounding_box
    cv2.rectangle(visualization, (adj_x1, adj_y1), (adj_x2, adj_y2), (0, 255, 0), 2)

    avg_color = np.mean(visualization, axis=(0, 1))
    brightness = np.sqrt(0.299 * (avg_color[0] ** 2) + 0.587 * (avg_color[1] ** 2) + 0.114 * (avg_color[2] ** 2))
    text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

    info_lines = [
        f"Method Used: {method_used}",
        f"Initial Bounding Box: {bounding_box}",
        f"Adjusted Bounding Box: {adjusted_bounding_box}",
        f"Crop Area: {crop_area_percentage:.2f}%",
        f"Aspect Ratio: {crop_details['actual_aspect_ratio']:.2f}",
        f"Padding Ratio: {crop_details['padding']}",
    ]
    y_position = 30
    for line in info_lines:
        cv2.putText(
            visualization, line, (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
        )
        y_position += 30

    return visualization


def product(image_source, image_path, output_directory=''):
    """Orchestrate the product-detection crop pipeline for a single image.
    image_source: an ImageSource instance — not a GCS bucket.
    EXPLORED AND NOT IN THE LIVE PATH: P-codes route to default crop via router.py."""
    try:
        # Load and process the image
        image_rgb = image_source.read(image_path)
        image_height, image_width, _ = image_rgb.shape
        print('Processing for product image done.')

        # Attempt to get bounding box with fallback strategy
        bounding_box, method_used = get_largest_product_box(image_rgb, image_height, image_width)
        print(f"Bounding box obtained using {method_used}: {bounding_box}")

        # Check if bounding_box has four values
        if not bounding_box or len(bounding_box) != 4:
            raise ValueError(f"No valid bounding box detected. Expected four coordinates, got: {bounding_box}")

        default_method = False
        if method_used == "Default Cropping":
            default_method = True

        # Crop the product image based on bounding box, padding, and aspect ratio adjustments
        cropped_image, crop_details = crop_product(image_rgb, image_height, image_width, bounding_box, default_method)
        print(f"Cropping completed using {method_used}")

        base_file_name = os.path.basename(image_path)
        if base_file_name.endswith('.jpg'):
            base_file_name = base_file_name[:-4]
        output_path_product = os.path.join(output_directory, f'{base_file_name}.jpg')

        return {'product': (cropped_image, output_path_product)}
    except ValueError as ve:
        return {'error': f"No product detected: {str(ve)}"}
    except Exception as e:
        return {'error': f"Error in product function: {str(e)}"}
