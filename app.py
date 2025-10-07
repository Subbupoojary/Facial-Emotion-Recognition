import cv2
from deepface import DeepFace
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Start capturing video
cap = cv2.VideoCapture(0)

# Global variable to store the latest frame for real-time streaming
current_frame = None

# Function to capture frames and perform emotion detection
def capture_frames():
    global current_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale to RGB (for face detection)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Store the latest frame
        current_frame = frame

# Run the frame capture function in a separate thread
threading.Thread(target=capture_frames, daemon=True).start()

# Function to generate the MJPEG stream
def generate():
    global current_frame
    while True:
        if current_frame is not None:
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame)
            if ret:
                # Yield the frame in MJPEG format for streaming
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Index route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
