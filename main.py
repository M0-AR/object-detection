import cv2
import numpy as np
import os

def load_image(filepath):
    """Loads an image and handles potential errors."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Failed to decode image at {filepath}")
    return image

def get_mock_detection():
    """
    Simulates a response from the Roboflow Object Detection API.
    In a production environment, this would call the Roboflow inference SDK.
    """
    return {
        'predictions': [
            {
                'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0,
                'confidence': 0.6979602575302124, 'class': 'window',
                'image_path': 'one-window.jpg'
            }
        ],
        'image': {'width': 460, 'height': 307}
    }

def apply_perspective_transform(src_image, dst_shape, points):
    """
    Applies perspective transformation to an asset to fit a target region.
    """
    src_h, src_w = src_image.shape[:2]
    src_points = np.float32([
        [0, 0],
        [src_w - 1, 0],
        [src_w - 1, src_h - 1],
        [0, src_h - 1]
    ])

    dst_points = np.float32(points)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the transformation
    transformed = cv2.warpPerspective(src_image, matrix, (dst_shape[1], dst_shape[0]))
    return transformed, dst_points

def blend_images(background, foreground, target_points):
    """
    Seamlessly blends a transformed foreground asset onto a background image
    using bitwise masking.
    """
    mask = np.zeros(background.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(target_points)], (255, 255, 255))
    mask_inverse = cv2.bitwise_not(mask)

    # Use the mask to blend the images
    bg_part = cv2.bitwise_and(background, mask_inverse)
    fg_part = cv2.bitwise_and(foreground, mask)

    return cv2.add(bg_part, fg_part)

def run_renovation_pipeline(base_img_path, asset_img_path, output_path):
    """Orchestrates the virtual renovation pipeline."""
    try:
        print(f"Starting pipeline for {base_img_path}...")

        # 1. Load Assets
        base_img = load_image(base_img_path)
        asset_img = load_image(asset_img_path)

        # 2. Get Object Detection Data (Mocked)
        detection = get_mock_detection()
        prediction = detection['predictions'][0]

        # 3. Calculate Target Coordinates
        x_c, y_c = prediction['x'], prediction['y']
        w, h = prediction['width'], prediction['height']

        # Define corner points for the window
        epsilon = 5 # Adjustment for perspective realism
        pts = [
            [x_c - w/2, y_c - h/2],
            [x_c + w/2, y_c - h/2 - epsilon],
            [x_c + w/2, y_c + h/2 - epsilon],
            [x_c - w/2, y_c + h/2 - epsilon]
        ]

        # 4. Transform Asset
        transformed_asset, target_pts = apply_perspective_transform(asset_img, base_img.shape, pts)

        # 5. Blend Images
        result = blend_images(base_img, transformed_asset, target_pts)

        # 6. Save Result
        cv2.imwrite(output_path, result)
        print(f"Successfully saved renovation to {output_path}")

    except Exception as e:
        print(f"Pipeline Error: {e}")

if __name__ == "__main__":
    run_renovation_pipeline(
        base_img_path='one-window.jpg',
        asset_img_path='midjourney.png',
        output_path='modified_image.jpg'
    )
