import time
import cv2
import numpy as np
from midjourney_api import generate_image_by_midjourney

def process_renovation():
    # 1. GENERATION STAGE
    print("Initiating Generative AI design phase...")
    try:
        generate_image_by_midjourney('modern window')
    except Exception as e:
        print(f"Midjourney API service unavailable: {e}. Using cached asset 'window.jpg'.")

    # 2. LOADING ASSETS
    original_image = cv2.imread('home.jpg')
    new_window_design = cv2.imread('window.jpg')

    if original_image is None or new_window_design is None:
        raise FileNotFoundError("Critical assets missing. Ensure home.jpg and window.jpg exist.")

    # 3. DETECTION STAGE (Mocked Roboflow Inference)
    # The system detects the window region in home.jpg
    response = {
        'predictions': [
            {
                'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0,
                'confidence': 0.6979, 'class': 'window', 'image_path': 'home.jpg'
            }
        ],
        'image': {'width': 460, 'height': 307}
    }

    # Extract spatial metadata
    pred = response['predictions'][0]
    x_center, y_center = int(pred['x']), int(pred['y'])
    width, height = int(pred['width']), int(pred['height'])

    # Calculate base corner points
    x1, y1 = x_center - width // 2, y_center - height // 2
    x2, y2 = x_center + width // 2, y_center - height // 2
    x3, y3 = x_center + width // 2, y_center + height // 2
    x4, y4 = x_center - width // 2, y_center + height // 2

    # 4. PERSPECTIVE TRANSFORMATION (The "Hero" Logic)
    # We define the source points (the flat AI-generated image)
    src_points = np.float32([
        [0, 0],
        [new_window_design.shape[1] - 1, 0],
        [new_window_design.shape[1] - 1, new_window_design.shape[0] - 1],
        [0, new_window_design.shape[0] - 1]
    ])

    # We define the destination points with perspective adjustment (Homography)
    # Adjusting right-side points slightly to simulate house angle/depth
    dst_points = np.float32([
        [x1, y1],           # Top Left
        [x2 + 2, y2 + 4],   # Top Right (adjusted for perspective)
        [x3 + 2, y3 - 2],   # Bottom Right (adjusted for perspective)
        [x4, y4]            # Bottom Left
    ])

    # Compute Homography Matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the generative design into the property perspective
    warped_design = cv2.warpPerspective(
        new_window_design, matrix, (original_image.shape[1], original_image.shape[0])
    )

    # 5. SEAMLESS BLENDING (Masking)
    # Create a binary mask for the transformation region
    mask = np.zeros(original_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
    mask_inverse = cv2.bitwise_not(mask)

    # Extract original background minus the window region
    background = cv2.bitwise_and(original_image, mask_inverse)

    # Extract warped design limited to the target mask
    foreground = cv2.bitwise_and(warped_design, mask)

    # Composite the final renovation
    final_image = cv2.add(background, foreground)

    # 6. OUTPUT STAGE
    cv2.imwrite('modified_image.jpg', final_image)
    print("Renovation complete. Output saved to modified_image.jpg")

if __name__ == "__main__":
    process_renovation()
