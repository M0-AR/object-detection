# https://towardsdatascience.com/stable-diffusion-as-an-api-5e381aec1f6
# import json
# import base64
#
# import requests
#
#
# def submit_post(url: str, data: dict):
#     """
#     Submit a POST request to the given URL with the given data.
#     """
#     return requests.post(url, data=json.dumps(data))
#
#
# def save_encoded_image(b64_image: str, output_path: str):
#     """
#     Save the given image to the given output path.
#     """
#     with open(output_path, "wb") as image_file:
#         image_file.write(base64.b64decode(b64_image))
#
#
# if __name__ == '__main__':
#     txt2img_url = 'http://127.0.0.1:7861/sdapi/v1/txt2img'
#     data = {'prompt': 'a dog wearing a hat'}
#     response = submit_post(txt2img_url, data)
#     save_encoded_image(response.json()['images'][0], 'dog.png')


import requests
from PIL import Image
from io import BytesIO


def submit_post(url, data):
    return requests.get(url, params=data)


def save_encoded_image(encoded_image, file_name):
    with open(file_name, 'wb') as f:
        f.write(encoded_image)


def generate_image_by_midjourney(prompt):
    txt2img_url = 'http://6a54-35-233-242-102.ngrok.io'
    data = {'prompt': prompt}
    response = submit_post(txt2img_url, data)
    save_encoded_image(response.content, 'midjourney.png')

"""
import cv2
import numpy as np
import time

# Load the original image and the new window design
original_image = cv2.imread('one-window.jpg')
new_window_design = cv2.imread('window.jpg')

time.sleep(2)

# JSON response
response = {
    'predictions': [
        {
            'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
        }
    ], 'image': {'width': '460', 'height': '307'}
}

# Extract the detected window coordinates (center) and size
x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])

# Calculate the top-left corner coordinates
x, y = x_center - width // 2, y_center - height // 2

# Resize the new window design to match the size of the detected window
resized_window_design = cv2.resize(new_window_design, (width, height))

# Apply perspective transformation to the resized window
pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts2 = np.float32([[0, 0], [width, 0], [int(0.5 * width), height], [int(1.5 * width), height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective_window_design = cv2.warpPerspective(resized_window_design, matrix, (2 * width, height))

# Overlay the transformed window image onto the original image with 100% opacity
window_mask = perspective_window_design[:, :, :3] != 0
original_image[y:y + height, x:x + 2 * width][window_mask] = perspective_window_design[window_mask]

# Save the modified image
cv2.imwrite('modified_image.jpg', original_image)

# Display the modified image (optional)
cv2.imshow('Modified Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""