from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained model
model = tf.keras.models.load_model("AtoZsign_language_model.h5")

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

@socketio.on("frame")
def handle_frame(data):
    # Decode the base64 image
    frame_data = data.split(",")[1]
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    # Detect hands and process the frame
    hands, _ = detector.findHands(frame)
    if hands:
        # Perform sign language detection
        img_input = preprocess_image(frame)
        predictions = model.predict(np.array([img_input]))
        predicted_class = np.argmax(predictions)
        emit("prediction", {"prediction": predicted_class})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)