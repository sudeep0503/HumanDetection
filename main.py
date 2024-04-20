# !wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
# !wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
# !wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
from google.colab import files

# Load YOLOv3 model
net = cv2.dnn.readNet("/content/yolov3.weights", "/content/yolov3.cfg")
classes = []

# Load class names
with open("/content/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Confidence threshold and NMS threshold for object detection
conf_threshold = 0.5
nms_threshold = 0.3

# Path to the input video file
video_path = "/content/vid2.mp4"

# Path to the folder where frames with detected persons will be saved
output_folder = "extracted_frames"
os.makedirs(output_folder, exist_ok=True)

# Open the video file for capturing frames
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video!")
    exit()

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a format suitable for input to the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Confidence scores for each class
            class_id = np.argmax(scores)  # ID of the class with the highest score
            confidence = scores[class_id]  # Confidence score for the detected class

            # Check if the detected object is a person and if the confidence is above the threshold
            if confidence > conf_threshold and classes[class_id] == "person":
                # Save the frame with the detected person to the output folder
                video_name, _ = os.path.splitext(os.path.basename(video_path))
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                filename = f"{output_folder}/{video_name}_frame_{frame_number}.jpg"
                cv2.imwrite(filename, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Create a zip file containing all the detected frames
!zip -r extracted_frames.zip {output_folder}

# Download the zip file
files.download("extracted_frames.zip")
