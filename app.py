import cv2
import numpy as np
import mediapipe as mp
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup pose detection with appropriate parameters
pose = mp_pose.Pose(
    static_image_mode=False,        # Process video stream
    model_complexity=1,             # 0, 1, or 2. Higher = more accurate but slower
    smooth_landmarks=True,          # Reduce jitter
    enable_segmentation=False,      # Optional: Background removal mask
    smooth_segmentation=True,       # Optional: Smooth the mask
    min_detection_confidence=0.5,   # Minimum confidence for person detection
    min_tracking_confidence=0.5     # Minimum confidence for landmark tracking
)

def calculate_angle(landmark1, landmark2, landmark3):
    """
    Calculate the angle between three landmarks (p1, p2, p3), where p2 is the vertex.
    Landmarks are expected as tuples (x, y, z).
    Returns angle in degrees (0-360).
    """
    # Check if landmarks are valid (basic check)
    if not landmark1 or not landmark2 or not landmark3:
        return 0 # Or raise an error, or return None

    # Get coordinates
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate angle using atan2
    # Angle of vector p2->p1
    angle1 = math.atan2(y1 - y2, x1 - x2)
    # Angle of vector p2->p3
    angle2 = math.atan2(y3 - y2, x3 - x2)

    # Calculate the difference -> angle at p2
    angle = math.degrees(angle2 - angle1)

    # Normalize angle to be within 0-360 degrees
    # This normalization keeps the angle relative to the vector p2->p1
    # if angle < 0:
    #    angle += 360
    # Alternative normalization: Ensure angle is always the smaller interior angle (0-180) - often more intuitive for poses
    angle = abs(angle)
    if angle > 180:
         angle = 360 - angle

    return angle

def classify_pose(landmarks):
    """
    Classify yoga pose based on joint angles and relative positions.
    Takes a list of landmarks where each landmark is (x, y, z).
    """
    label = "Unknown Pose"

    # Check if we have enough landmarks
    if len(landmarks) < max(mp_pose.PoseLandmark): # Basic check
         return label

    # --- Calculate all relevant angles first ---
    try:
        # Get landmark coordinates using mp_pose.PoseLandmark enum values
        # Left Arm
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Right Arm
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Left Leg
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # Right Leg
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate Elbow Angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Calculate Shoulder Angles (arm relative to torso)
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

        # Calculate Knee Angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate Hip Angles (leg relative to torso)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # --- Pose Classification Logic ---

        # Get Y coordinates for vertical position checks (lower Y means higher up in image)
        shoulder_y_avg = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y_avg = (left_hip[1] + right_hip[1]) / 2
        knee_y_avg = (left_knee[1] + right_knee[1]) / 2
        ankle_y_avg = (left_ankle[1] + right_ankle[1]) / 2


        # ** T Pose **
        # Arms straight out, legs straight down
        if (160 < left_elbow_angle < 200 and 160 < right_elbow_angle < 200 and
            75 < left_shoulder_angle < 115 and 75 < right_shoulder_angle < 115 and
            160 < left_knee_angle < 200 and 160 < right_knee_angle < 200 and
            160 < left_hip_angle < 200 and 160 < right_hip_angle < 200): # Hips relatively straight too
            label = "T Pose"

        # ** Warrior II Pose **
        # Arms like T-Pose, one leg bent forward, one leg straight back
        elif (160 < left_elbow_angle < 200 and 160 < right_elbow_angle < 200 and
              75 < left_shoulder_angle < 115 and 75 < right_shoulder_angle < 115):
            # Check for one knee bent (~90 deg) and one knee straight (~180 deg)
            if ((80 < left_knee_angle < 130 and 160 < right_knee_angle < 200) or
                (80 < right_knee_angle < 130 and 160 < left_knee_angle < 200)):
                 label = "Warrior Pose" # (Warrior II)

        # ** Tree Pose **
        # Standing on one leg, the other foot resting on the standing leg. Arms can vary.
        # Check one leg straight, other knee bent outwards significantly
        elif ((160 < left_knee_angle < 200 and 30 < right_knee_angle < 120 and right_hip_angle < 150) or # Standing on Left
              (160 < right_knee_angle < 200 and 30 < left_knee_angle < 120 and left_hip_angle < 150)):   # Standing on Right
                 # Optional: Add arm conditions if desired (e.g., hands in prayer near chest, arms straight up)
                 label = "Tree Pose"


        # ** Downward Dog **
        # Body like inverted V. Hands and feet on floor.
        # Arms straight, legs straight, hips high, sharp angle at hips and shoulders.
        elif (160 < left_elbow_angle < 200 and 160 < right_elbow_angle < 200 and   # Arms straight
              160 < left_knee_angle < 200 and 160 < right_knee_angle < 200 and     # Legs straight
              45 < left_hip_angle < 95 and 45 < right_hip_angle < 95 and         # Sharp hip angle (body-leg)
              150 < left_shoulder_angle < 190 and 150 < right_shoulder_angle < 190 and # Wide shoulder angle (arm-body)
               hip_y_avg < knee_y_avg ): # Hips higher than knees
                 label = "Downward Dog"

        # ** Plank Pose **
        # Body straight line, supported on hands or forearms.
        # Knees, Hips, Shoulders form a near-straight line relative to body axis.
        elif (150 < left_knee_angle < 190 and 150 < right_knee_angle < 190 and    # Knees straight
              150 < left_hip_angle < 190 and 150 < right_hip_angle < 190 and      # Hips straight
              70 < left_shoulder_angle < 110 and 70 < right_shoulder_angle < 110):  # Shoulders ~90 deg to torso
            # Check elbows for high vs forearm plank
            if (150 < left_elbow_angle < 190 and 150 < right_elbow_angle < 190):
                label = "Plank Pose (High)"
            elif (70 < left_elbow_angle < 110 and 70 < right_elbow_angle < 110):
                 label = "Plank Pose (Forearm)"
            else:
                 label = "Plank Pose" # Generic if arms ambiguous

        # ** Cobra Pose **
        # Chest lifted, hips on floor, elbows potentially bent.
        elif (hip_y_avg > shoulder_y_avg and # Hips lower than shoulders (on/near floor)
              150 < left_knee_angle < 190 and 150 < right_knee_angle < 190): # Legs mostly straight
              # Elbows can be bent or straight depending on variation
              if (100 < left_elbow_angle < 190 and 100 < right_elbow_angle < 190):
                 label = "Rest position"

        # ** Bridge Pose **
        # Lying supine, hips lifted high, knees bent, feet flat.
        elif (hip_y_avg < shoulder_y_avg and hip_y_avg < knee_y_avg and # Hips are the highest point
              60 < left_knee_angle < 110 and 60 < right_knee_angle < 110): # Knees bent significantly
              # Shoulders should be relatively low compared to hips
              # Optional: Check shoulder angles if needed (likely > 90)
              label = "Bridge Pose"

        # ** Triangle Pose **
        # Legs wide and straight, torso hinged sideways, arms in a line.
        elif (155 < left_knee_angle < 195 and 155 < right_knee_angle < 195): # Both legs straight
             # Check for torso hinge: one hip angle acute, other obtuse relative to torso axis (tricky)
             # Check for shoulder angles indicating arms are out/aligned vertically
             # Simplified check: Assume straight legs and one arm pointing down, one up
             # Requires knowing which side is hinged - hard without orientation
             # Let's try checking shoulder abduction close to 90
             if ((60 < left_shoulder_angle < 120 or 60 < right_shoulder_angle < 120) and
                 abs(left_shoulder[1] - right_shoulder[1]) > abs(left_shoulder[0] - right_shoulder[0]) * 0.5): # Arms roughly vertical?
                 # Add hip hinge check if possible - e.g., angle between spine and one leg
                 label = "Triangle Pose" # Needs refinement

        # ** Boat Pose **
        # Seated V-shape, legs lifted (straight or bent).
        elif (45 < left_hip_angle < 110 and 45 < right_hip_angle < 110 and # Torso-leg angle (V-shape)
              hip_y_avg > ankle_y_avg and hip_y_avg > knee_y_avg ): # Feet/knees off the ground and higher than hips
             # Knees can be straight or bent
             # Optional: check arms straight forward (shoulder angle ~90, elbow angle ~180)
             label = "Boat Pose"

        # ** Bow Pose **
        # Prone, grabbing ankles, lifting chest and thighs. Very arched back.
        elif (hip_y_avg > shoulder_y_avg and # Hips lower than shoulders
              15 < left_knee_angle < 90 and 15 < right_knee_angle < 90): # Knees very bent
              # Check if wrists are near ankles (requires coordinate comparison, adding complexity)
              wrist_l_y, wrist_r_y = left_wrist[1], right_wrist[1]
              ankle_l_y, ankle_r_y = left_ankle[1], right_ankle[1]
              # Basic check: shoulders and knees higher than hips?
              if shoulder_y_avg < hip_y_avg and knee_y_avg < hip_y_avg:
                   # Add wrist-ankle proximity check for robustness if needed
                   label = "Bow Pose"

        # ** Half Moon Pose **
        # Standing on one leg, other leg extended back, torso hinged, arms in line.
        # Very complex geometry
        elif ((160 < left_knee_angle < 200 and 160 < right_knee_angle < 200) and # Both legs straight
               (hip_y_avg < ankle_y_avg) ): # Check if at least one ankle is higher than hips (lifted leg)
                # Need more specific checks: one leg vertical, other horizontal, torso tilt, arm alignment
                # Example check: one hip angle ~90 (standing leg), other ~180 (lifted leg)
                # This is hard to get right reliably without more checks
                # Check if shoulders are roughly vertically aligned
                if abs(left_shoulder[0] - right_shoulder[0]) < abs(left_shoulder[1] - right_shoulder[1]) * 0.5: # Basic vertical check
                     label = "Half Moon Pose" # Needs significant refinement

        # ** Eagle Pose **
        # Standing, legs and arms crossed. Extremely complex angles.
        # Check for bent knees and elbows, potentially crossed limbs.
        elif (30 < left_knee_angle < 130 and 30 < right_knee_angle < 130 and   # Both knees bent
              30 < left_elbow_angle < 130 and 30 < right_elbow_angle < 130):   # Both elbows bent
             # This is a very weak condition. Need to check crossing.
             # e.g., is left knee x > right knee x and vice versa? Is left elbow x > right elbow x?
             # Check relative positions of knees/ankles and elbows/wrists
             # Simplified: If both knees and elbows are significantly bent, maybe it's Eagle.
             label = "Eagle Pose" # Needs significant refinement


    except Exception as e:
        # print(f"Error calculating angles or classifying pose: {e}")
        # Pass silently if a landmark wasn't detected properly for angle calculation
        pass # Keep label as "Unknown Pose"

    return label


def detect_pose(image):
    """
    Detect pose landmarks and classify the pose.
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Improve performance

    # Process the image and detect poses
    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True # Make image writeable again
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for drawing

    # Initialize list for landmarks [(x, y, z), ...]
    landmarks_list = []
    pose_name = "Unknown Pose"

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Get image dimensions
        height, width, _ = image.shape

        # Extract landmarks into a more accessible list format
        landmarks_list = [(int(lm.x * width), int(lm.y * height), lm.z * width) # Use width for Z scaling as proxy
                          for lm in results.pose_landmarks.landmark]

        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Classify the pose using the extracted landmarks list
        if landmarks_list: # Ensure landmarks were extracted
             pose_name = classify_pose(landmarks_list)

        # Draw the pose classification on the image
        cv2.putText(image, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Return image, pose name, and the raw landmarks if needed elsewhere
        return image, pose_name, results.pose_landmarks # Return raw mediapipe landmarks if needed

    # Return original image and default values if no landmarks detected
    return image, "No Pose Detected", None # Indicate no pose found

@app.route('/detect_pose', methods=['POST'])
def process_image():
    try:
        # Check if image file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
             return jsonify({'error': 'No selected file'}), 400

        # Read image file
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
             return jsonify({'error': 'Could not decode image'}), 400

        # Detect pose
        processed_img, pose_name, _ = detect_pose(img) # We don't need landmarks in the response here

        # Convert processed image to base64 for sending back to client
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Return the results
        return jsonify({
            'pose': pose_name,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        print(f"Server Error: {e}") # Log error server-side
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Use debug=False in production