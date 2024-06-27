from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Global variables to hold the name and face data
file_name = ""
face_data = []
capture = False
proctor = False
cap = None  # Global variable for VideoCapture

# Function to capture data
def run_capture(frame):
    global face_data, file_name

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) == 0:
        return frame

    k = 1
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    for face in faces[:1]:
        x, y, w, h = face
        offset = 5
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_selection = cv2.resize(face_offset, (100, 100))

        if len(face_data) % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Function for test proctoring
def run_test(frame):
    global proctor

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    dataset_path = "./face_dataset/"
    face_data = []
    labels = []
    class_id = 0
    names = {}

    # Dataset preparation
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(os.path.join(dataset_path, fx))
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

    trainset = np.concatenate((face_dataset, face_labels), axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    if len(faces) == 0:
        cv2.putText(frame, "Alert: No face detected!", (20, 50), font, 1, (0, 0, 255), 2)
    elif len(faces) > 1:
        cv2.putText(frame, "Alert: Multiple faces detected!", (20, 50), font, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            face_section = frame[y-5:y+h+5, x-5:x+w+5]
            face_section = cv2.resize(face_section, (100, 100))

            out = knn(trainset, face_section.flatten())

            cv2.putText(frame, names[int(out)], (x, y-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    return frame

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global file_name, capture, face_data, cap
    file_name = request.form['name']
    face_data = []
    capture = True
    cap = cv2.VideoCapture(0)  # Start capturing video
    return redirect(url_for('index'))

@app.route('/start_proctoring', methods=['POST'])
def start_proctoring():
    global proctor, cap
    proctor = True
    return redirect(url_for('index'))

@app.route('/stop_proctoring', methods=['POST'])
def stop_proctoring():
    global proctor, cap
    proctor = False
    if cap:
        cap.release()  # Release the camera
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global capture, proctor, face_data, file_name, cap
    while True:
        if cap and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame if camera not available

        if capture:
            frame = run_capture(frame)
            if len(face_data) >= 100:
                face_data = np.array(face_data)
                face_data = face_data.reshape((face_data.shape[0], -1))
                np.save("./face_dataset/" + file_name, face_data)
                capture = False

        if proctor:
            frame = run_test(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/static/<path:path>')
def static_file(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    if not os.path.exists('face_dataset'):
        os.makedirs('face_dataset')
    app.run(debug=True)
