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
from midjourney_api import generate_image_by_midjourney

# Load the original image and the new window design
original_image = cv2.imread('one-window.jpg')
generate_image_by_midjourney('just white window  width: 62.0 height: 82.0')

time.sleep(2)
new_window_design = cv2.imread('midjourney.png')

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
