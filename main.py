# import cv2
# import numpy as np
# import time

# # Function to detect eyes using Haar cascades
# def detect_eyes(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     return eyes

# # Initialize the video capture object and variables
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     eyes = detect_eyes(frame)
    
#     # Print debug information
#     print("Detected Eyes:", eyes)

#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     cv2.imshow("Eye Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to determine if the eyes are looking left or right
def determine_eye_direction(left_eye_center, right_eye_center, frame_center):
    eye_midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
    dx = eye_midpoint[0] - frame_center[0]
    dy = eye_midpoint[1] - frame_center[1]
    
    if abs(dx) > abs(dy):  # Horizontal movement is greater than vertical
        if dx > 0:
            return "Right"
        else:
            return "Left"
    else:  # Vertical movement is greater than horizontal
        if dy > 0:
            return "Down"
        else:
            return "Up"

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Counters for eye directions
left_counter = 0
right_counter = 0


direction = None
direction_count = 0
direction_threshold = 30 
cheating_threshold = 300

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
           
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = [face_landmarks.landmark[i] for i in left_eye_indices]
            right_eye = [face_landmarks.landmark[i] for i in right_eye_indices]

           
            left_eye_center = np.mean([(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in left_eye], axis=0).astype(int)
            right_eye_center = np.mean([(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in right_eye], axis=0).astype(int)

          
            new_direction = determine_eye_direction(left_eye_center, right_eye_center, frame_center)
            
            
            if new_direction == direction:
                direction_count += 1
            else:
                direction = new_direction
                direction_count = 1
            
            # Update counters
            if new_direction == "Left":
                left_counter += 1
            elif new_direction == "Right":
                right_counter += 1
            
            # Draw eye centers and direction text on the frame
            cv2.circle(frame, tuple(left_eye_center), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 2, (0, 255, 0), -1)
            cv2.putText(frame, f"Direction: {new_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if direction count exceeds threshold for cheating alert
            if direction_count > direction_threshold and (new_direction == "Left" or new_direction == "Right") \
               and (left_counter > cheating_threshold or right_counter > cheating_threshold):
                cv2.putText(frame, "Alert: Cheating Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Alert: Cheating Detected!")

    cv2.imshow("Eye Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Looked Left: {left_counter} times")
print(f"Looked Right: {right_counter} times")
