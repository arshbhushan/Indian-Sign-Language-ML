from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from sklearn.preprocessing import LabelEncoder
import dlib
from scipy.spatial import distance as dist
import time
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model("AtoZsign_language_model.h5")

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Initialize dlib's face detector and facial landmark predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for EAR-based blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Define the list of classes
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Decode the base64 image from the frontend
    data = request.json
    image_data = data['image'].split(",")[1]  # Remove the data URL prefix
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        # Process hands and predict
        # (Add the logic from your Python script here)
        # For brevity, I'm skipping the full implementation.

        # Example: Return a dummy prediction
        prediction = "A"
        return jsonify({"prediction": prediction})
    else:
        return jsonify({"error": "No hands detected"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)