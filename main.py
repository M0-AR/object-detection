# Replace os.getenv("HOME") with os.getenv("USERPROFILE") for Windows.
# from roboflow import Roboflow
#
# rf = Roboflow(api_key="")
# project = rf.workspace().project("windows-instance-segmentation")
# model = project.version(3).model
#
# # infer on a local image
# print(model.predict("home.jpg").json())
#
# # infer on an image hosted elsewhere
# # print(model.predict("URL_OF_YOUR_IMAGE").json())
#
# # save an image annotated with your predictions
# model.predict("home.jpg").save("prediction.jpg")

# Replace os.getenv("HOME") with os.getenv("USERPROFILE") for Windows.
"""
from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace().project("window-detector")
model = project.version(2).model

# infer on a local image
print(model.predict("one-window.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("one-window.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
"""
import time
import cv2
import numpy as np

"""

from midjourney_api import generate_image_by_midjourney
# generate_image_by_midjourney('window')

# Load the original image and the new window design
original_image = cv2.imread('one-window.jpg')

time.sleep(2)
new_window_design = cv2.imread('window.jpg')

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

# Replace the detected window in the original image with the resized new window design
original_image[y:y+height, x:x+width] = resized_window_design

# Save the modified image
cv2.imwrite('modified_image.jpg', original_image)

# Display the modified image (optional)
cv2.imshow('Modified Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
# import cv2
# import numpy as np
# import time
#
# # Load the original image and the new window design
# original_image = cv2.imread('one-window.jpg')
# new_window_design = cv2.imread('window.jpg')
#
# time.sleep(2)
#
# # JSON response
# response = {
#     'predictions': [
#         {
#             'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
#         }
#     ], 'image': {'width': '460', 'height': '307'}
# }
#
# # Extract the detected window coordinates (center) and size
# x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])
#
# # Calculate the top-left corner coordinates
# x, y = x_center - width // 2, y_center - height // 2
#
# # Resize the new window design to match the size of the detected window
# resized_window_design = cv2.resize(new_window_design, (width, height))
#
# # Apply perspective transformation to the resized window
# pts1 = np.float32([[0, 0], [width, 0], [int(0.5 * width), height], [int(1.1 * width), height]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# perspective_window_design = cv2.warpPerspective(resized_window_design, matrix, (2 * width, height))
#
# # Overlay the transformed window image onto the original image with 100% opacity
# window_mask = perspective_window_design[:, :, :3] != 0
# original_image[y:y + height, x:x + 2 * width][window_mask] = perspective_window_design[window_mask]
#
# # Save the modified image
# cv2.imwrite('modified_image.jpg', original_image)
#
# # Display the modified image (optional)
# cv2.imshow('Modified Image', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import time
#
# # Load the original image and the new window design
# original_image = cv2.imread('one-window.jpg')
# new_window_design = cv2.imread('window.jpg')
#
# time.sleep(2)
#
# # JSON response
# response = {
#     'predictions': [
#         {
#             'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
#         }
#     ], 'image': {'width': '460', 'height': '307'}
# }
#
# # Extract the detected window coordinates (center) and size
# x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])
#
# # Calculate the top-left corner coordinates
# x, y = x_center - width // 2, y_center - height // 2
#
# # Resize the new window design to match the size of the detected window
# resized_window_design = cv2.resize(new_window_design, (width, height))
#
# # Apply perspective transformation to the resized window
# pts1 = np.float32([[0, 0], [width, 0], [int(0.4 * width), height], [int(1.5 * width), height]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
#
# perspective_window_design = cv2.warpPerspective(resized_window_design, matrix, (2 * width, height))
#
# # Overlay the transformed window image onto the original image with 100% opacity
# window_mask = perspective_window_design[:, :, :3] != 0
# original_image[y:y + height, x:x + 2 * width][window_mask] = perspective_window_design[window_mask]
#
# # Save the modified image
# cv2.imwrite('modified_image.jpg', original_image)
#
# # Display the modified image (optional)
# cv2.imshow('Modified Image', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import time
import cv2
import numpy as np
import math

# # Load the original image and the new window design
# original_image = cv2.imread('one-window.jpg')
#
# time.sleep(2)
# new_window_design = cv2.imread('window.jpg')
#
# # JSON response
# response = {
# 'predictions': [
# {
# 'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
# }
# ], 'image': {'width': '460', 'height': '307'}
# }
#
# # Extract the detected window coordinates (center) and size
# x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])
#
# # Calculate the top-left corner coordinates
# x, y = x_center - width // 2, y_center - height // 2
#
# orientation_degree = 0
#
# # Rotate the new window design by the given degree
# (h, w) = new_window_design.shape[:2]
# (centerX, centerY) = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D((centerX, centerY), orientation_degree, 1.0)
# rotated_window_design = cv2.warpAffine(new_window_design, M, (w, h))
#
# # Resize the rotated window design to match the size of the detected window
# resized_window_design = cv2.resize(rotated_window_design, (width, height))
#
# # Replace the detected window in the original image with the resized and rotated new window design
# original_image[y:y+height, x:x+width] = resized_window_design
#
# # Save the modified image
# cv2.imwrite('modified_image.jpg', original_image)
#
# # Display the modified image (optional)
# cv2.imshow('Modified Image', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import time
# import cv2
# import numpy as np
#
# # Load the original image and the new window design
# original_image = cv2.imread('one-window.jpg')
# # new_window_design = cv2.imread('midjourney.png')
# new_window_design = cv2.imread('window.jpg')
#
# # JSON response: This response coming from Roboflow api
# response = {
#     'predictions': [
#         {
#             'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
#         }
#     ], 'image': {'width': '460', 'height': '307'}
# }
#
# width = int(response['image']['width'])
# height = int(response['image']['height'])
#
# x_c = response['predictions'][0]['x'] * float(response['image']['width']) / width
# y_c = response['predictions'][0]['y'] * float(response['image']['height']) / height
# w = response['predictions'][0]['width'] * float(response['image']['width']) / width
# h = response['predictions'][0]['height'] * float(response['image']['height']) / height
#
# w_half = w / 2
# h_half = h / 2
#
# x1 = int(x_c - w_half)
# y1 = int(y_c - h_half)
# x2 = int(x_c + w_half)
# y2 = int(y_c - h_half)
# x3 = int(x_c + w_half)
# y3 = int(y_c + h_half)
# x4 = int(x_c - w_half)
# y4 = int(y_c + h_half)
#
#
# # Extract the detected window coordinates (center) and size
# x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])
#
# # Calculate the corner points of the window region
# x1, y1 = x_center - width // 2, y_center - height // 2
# x2, y2 = x_center + width // 2, y_center - height // 2
# x3, y3 = x_center + width // 2, y_center + height // 2
# x4, y4 = x_center - width // 2, y_center + height // 2
#
#
# # x1, y1 = 167,120
# # x2, y2 = 229,113
# # x3, y3 = 229,195
# # x4, y4 = 167,199
#
# # Define points in the source image (window image)
# src_points = np.float32([[0, 0], [new_window_design.shape[1] - 1, 0], [new_window_design.shape[1] - 1, new_window_design.shape[0] - 1], [0, new_window_design.shape[0] - 1]])
#
# # Define points in the destination image (original image)
# dst_points = np.float32([[x1, y1], [x2, y2 - 8], [x3, y3 - 8], [x4, y4 - 2]])
#
# # Compute the homography transformation matrix
# H, _ = cv2.findHomography(src_points, dst_points)
#
# # Apply the transformation to the window image
# new_window_transformed = cv2.warpPerspective(new_window_design, H, (original_image.shape[1], original_image.shape[0]))
#
# # Create a mask to overlay the new window on the original image
# mask = np.zeros(original_image.shape, dtype=np.uint8)
# cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
# mask_inverse = cv2.bitwise_not(mask)
#
# # Use the mask to blend the original image and the transformed window image
# original_image = cv2.bitwise_and(original_image, mask_inverse)
# new_window_transformed = cv2.bitwise_and(new_window_transformed, mask)
# final_image = cv2.add(original_image, new_window_transformed)
#
# # Show the updated original image with the replaced window
# cv2.imshow('Modified Image', final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import time
import cv2
import numpy as np

# Load the original image and the new window design
original_image = cv2.imread('home.jpg')
new_window_design = cv2.imread('window.jpg')

# JSON response: This response coming from Roboflow api
response = {
    'predictions': [
        {
            'x': 198.0, 'y': 161.0, 'width': 62.0, 'height': 82.0, 'confidence': 0.6979602575302124, 'class': 'window', 'image_path': 'one-window.jpg', 'prediction_type': 'ObjectDetectionModel'
        }
    ], 'image': {'width': '460', 'height': '307'}
}

# Extract the detected window coordinates (center) and size
x_center, y_center, width, height = int(response['predictions'][0]['x']), int(response['predictions'][0]['y']), int(response['predictions'][0]['width']), int(response['predictions'][0]['height'])

# Calculate the corner points of the window region
x1, y1 = x_center - width // 2, y_center - height // 2
x2, y2 = x_center + width // 2, y_center - height // 2
x3, y3 = x_center + width // 2, y_center + height // 2
x4, y4 = x_center - width // 2, y_center + height // 2
#
# Define points in the source image (window image)
src_points = np.float32([[0, 0], [new_window_design.shape[1] - 1, 0], [new_window_design.shape[1] - 1, new_window_design.shape[0] - 1], [0, new_window_design.shape[0] - 1]])

# Define points in the destination image (original image)
epsilon = 5
dst_points = np.float32([[x1, y1], [x2, y2 - epsilon], [x3, y3 - epsilon], [x4, y4 - epsilon]])

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)
# Apply the transformation to the window image
new_window_transformed = cv2.warpPerspective(new_window_design, M, (original_image.shape[1], original_image.shape[0]))

# Compute the homography transformation matrix
# H, _ = cv2.findHomography(src_points, dst_points)
# Apply the transformation to the window image
# new_window_transformed = cv2.warpPerspective(new_window_design, H, (original_image.shape[1], original_image.shape[0]))

# Create a mask to overlay the new window on the original image
mask = np.zeros(original_image.shape, dtype=np.uint8)
cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
mask_inverse = cv2.bitwise_not(mask)

# Use the mask to blend the original image and transformed window image
original_image = cv2.bitwise_and(original_image, mask_inverse)
new_window_transformed = cv2.bitwise_and(new_window_transformed, mask)
final_image = cv2.add(original_image, new_window_transformed)

# Show the updated original image with the replaced window
cv2.imshow('Modified Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
