# import cv2

# # Function to extract keypoints and descriptors using ORB
# def extract_keypoints_and_descriptors(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Initialize ORB detector
#     orb = cv2.ORB_create()
#     # Detect keypoints and compute descriptors
#     keypoints, descriptors = orb.detectAndCompute(gray, None)
#     return keypoints, descriptors

# # Function to match descriptors using BFMatcher
# def match_descriptors(descriptors1, descriptors2):
#     # Initialize BFMatcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # Match descriptors
#     matches = bf.match(descriptors1, descriptors2)
#     # Sort matches based on distance
#     matches = sorted(matches, key=lambda x: x.distance)
#     return matches

# # Path to the stored image
# stored_image_path = 'me.jpeg'

# # Load the stored image
# stored_image = cv2.imread(stored_image_path)
# if stored_image is None:
#     print(f"Error: Unable to open image at {stored_image_path}")
#     exit()

# # Extract keypoints and descriptors from the stored image
# stored_keypoints, stored_descriptors = extract_keypoints_and_descriptors(stored_image)

# # Initialize webcam
# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         continue

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         face_offset = frame[y:y+h, x:x+w]
        
#         # Extract keypoints and descriptors from the detected face
#         keypoints, descriptors = extract_keypoints_and_descriptors(face_offset)

#         if descriptors is not None and stored_descriptors is not None:
#             # Match descriptors between stored image and detected face
#             matches = match_descriptors(stored_descriptors, descriptors)
            
            
#             match_ratio = len(matches) / len(stored_descriptors)
            
#             if match_ratio > 0.1:  
#                 label = "Match"
#                 color = (0, 255, 0)
#             else:
#                 label = "No Match"
#                 color = (0, 0, 255)
#         else:
#             label = "No Match"
#             color = (0, 0, 255)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     cv2.imshow("faces", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2

def extract_keypoints_and_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_descriptors(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

stored_image_path = 'clgid.jpeg'
stored_image = cv2.imread(stored_image_path)
if stored_image is None:
    print(f"Error: Unable to open image at {stored_image_path}")
    exit()

stored_image_resized = cv2.resize(stored_image, (200, 200))
stored_keypoints, stored_descriptors = extract_keypoints_and_descriptors(stored_image_resized)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(300, 300))

    for (x, y, w, h) in faces:
        face_offset = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_offset, (200, 200))
        keypoints, descriptors = extract_keypoints_and_descriptors(face_resized)

        if descriptors is not None and stored_descriptors is not None:
            matches = match_descriptors(stored_descriptors, descriptors)
            match_ratio = len(matches) / min(len(stored_descriptors), len(descriptors))

            if match_ratio > 0.3:
                label = "Match"
                color = (0, 255, 0)
            else:
                label = "No Match"
                color = (0, 0, 255)
        else:
            label = "No Match"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()