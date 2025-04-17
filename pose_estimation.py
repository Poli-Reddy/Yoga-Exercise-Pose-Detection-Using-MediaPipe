#!/bin/bash

# Install matplotlib-venn
!pip install matplotlib-venn

# Install fluidsynth (alternative to libfluidsynth1)
!pip install pyfluidsynth || apt-get install -y fluidsynth

# Install libarchive
!pip install libarchive-c || (apt-get update && apt-get install -y libarchive-dev)

# Verify installation

# Install graphviz and pydot
!pip install cartopy
!pip install pydot

# Import pydot
import pydot

# Install cartopy
!pip install cartopy

# Import cartopy
import cartopy

   !pip install mediapipe
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3,model_complexity=2)
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils
sample_img=cv2.imread("x.jpg")
plt.figure(figsize=[10,10])
plt.title("sample_img");
plt.axis('off');
plt.imshow(sample_img[:,:,::-1]);
plt.show()
# Perform pose detection after converting the image into RGB format.
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

# Check if any landmarks are found.
if results.pose_landmarks:
    # Iterate two times as only want to display first two landmarks.
    for i in range(33):
        # Display the found normalized landmarks.
        print(f"{mp_pose.PoseLandmark(i).name}: {results.pose_landmarks.landmark[i].x}, {results.pose_landmarks.landmark[i].y}, {results.pose_landmarks.landmark[i].z}")
# Retrieve the height and width of the sample image.
image_height, image_width, _ = sample_img.shape

# Check if any landmarks are found.
if results.pose_landmarks:
    # Iterate two times as we only want to display the first two landmarks.
    for i in range(33):
        landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]

        print(f'{mp_pose.PoseLandmark(i).name}:')
        print(f'x: {landmark.x * image_width}')
        print(f'y: {landmark.y * image_height}')
        print(f'z: {landmark.z * image_width}')
        print(f'visibility: {landmark.visibility}\n')

# Create a copy of the sample image to draw landmarks on.
img_copy = sample_img.copy()

# Check if any landmarks are found.
if results.pose_landmarks:
    # Draw Pose Landmarks on the sample image.
    mp_drawing.draw_landmarks(
        image=img_copy,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS
    )

    # Specify a size of the figure.
    fig = plt.figure(figsize=[10, 10])

    # Display the output image with the landmarks drawn, also convert BGR to RGB for display.
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.show()
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image,
                                  landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

        if display:
            # Display the original input image and the resultant image.
            plt.figure(figsize=(10, 5))

            # Plot the original image
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
            plt.title("Original Image")
            plt.axis("off")

            # Plot the resultant image
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title("Resultant Image")
            plt.axis("off")

            plt.show()
            mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)

    # Return the output image and the found landmarks.
    return output_image, landmarks

import cv2
image = cv2.imread('bow.jpg')
detectPose(image, pose,display=True)
image = cv2.imread("cobra.jpg")
detectPose(image, pose,display=True)
image = cv2.imread("Tree Pose.jpeg")
detectPose(image, pose,display=True)
image =cv2.imread("halfwheel.jpg")
detectPose(image, pose,display=True)
image =cv2.imread("Warrior II.jpeg")
detectPose(image, pose,display=True)
image =cv2.imread("me.jpg")
detectPose(image, pose,display=True)
image =cv2.imread("T pose.jpeg")
detectPose(image, pose,display=True)
import cv2
import mediapipe as mp
import time

# Import the Pose module from MediaPipe.
mp_pose = mp.solutions.pose

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(1)
video.set(3, 1280)
video.set(4, 960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Initialize the time variable.
time1 = time.time()

# Iterate until the webcam is accessed successfully.
while video.isOpened():
    # Read a frame.
    ok, frame = video.read()

    # Check if frame is not read properly.
    if not ok:
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame.
    frame_height, frame_width, _ = frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Perform Pose Landmark detection.
    frame, _ = detectPose(frame, pose_video, display=False)
    time2 = time.time()

    # Check if the landmarks are detected.
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(frame, 'FPS:{}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

    time1 = time2
    cv2.imshow('Pose Detection', frame)

    # Wait until a key is pressed.
    # Retrieve the ASCII code of the key pressed.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if k == 27:
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
video.release()
cv2.destroyAllWindows()

def calculateAngle(landmark1, landmark2, landmark3):
    """
    This function calculates the angle between three different landmarks.

    Args:
        landmark1: The first landmark containing the x, y, and z coordinates.
        landmark2: The second landmark containing the x, y, and z coordinates.
        landmark3: The third landmark containing the x, y, and z coordinates.

    Returns:
        angle: The calculated angle between the three landmarks.
    """
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Normalize the angle to be within 0 to 360 degrees
    if angle < 0:
        angle += 360

    # Return the calculated angle
    return angle

# Calculate the angle between the three landmarks.
angle = calculateAngle((558, 326, 0), (642, 333, 0), (718, 321, 0))
print(f'The calculated angle is {angle}')



def classifyPose(landmarks, output_image, display=False):
    """
    Classifies yoga poses based on angles of various body joints.

    Args:
        landmarks: A list of detected landmarks from a person's pose.
        output_image: An image of the person with detected pose landmarks.
        display: If True, displays the image with the pose label.

    Returns:
        output_image: The image with pose label written on it.
        label: The classified pose label.
    """

    # Default pose label
    label = "Unknown Pose"
    color = (0, 0, 255)  # Red color for unknown pose

    # Calculate required angles
    left_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )
    right_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )
    left_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    )
    right_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    )
    left_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    )
    right_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    )

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

            # Check if it is the warrior II pose.
            # --------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'

    # ----------------------------------------------------------------

    # Check if it is the T pose.
    # --------------------------------------------------------------

        # Check if both legs are straight
        if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

            # Specify the label of the pose that is T pose.
            label = 'T Pose'

    # ----------------------------------------------------------------

    # Check if it is the tree pose.
    # --------------------------------------------------------------

    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'

    # --------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Unknown Pose':

        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

    # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        plt.show()

    # Ensure the function always returns a tuple
    return output_image, label

image=cv2.imread('Tree Pose.jpeg')
output_image,landmarks=detectPose(image,pose,display=False)
if landmarks:
    classifyPose(landmarks,output_image,display=True)
import cv2
import mediapipe as mp
import time

# Import the Pose module from MediaPipe.
mp_pose = mp.solutions.pose

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly.
    if not ok:

        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

# Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

# Perform Pose Landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)

# Check if the landmarks are detected.
    if landmarks:

    #  Perform the Pose Classification.
       frame, _ = classifyPose(landmarks, frame, display=False)

# Display the frame.
    cv2.imshow('Pose Classification', frame)

# Wait until a key is pressed.
# Retrieve the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
# Check if 'ESC' is pressed.
    if(k == 27):

    # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()






