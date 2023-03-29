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

