from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import dlib
from scipy.spatial import distance as dist

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model("AtoZsign_language_model.h5")

# Define the list of classes (in the same order as during training)
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# Initialize dlib's face detector and facial landmark predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for EAR-based blink detection
EYE_AR_THRESH = 0.25  # EAR threshold to consider a blink
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm a blink

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

# Function to get word suggestions
def get_word_suggestions(alphabet):
    word_dict = {
        "A": ["are", "apple", "A"],
        "B": ["be", "ball", "boy"],
        "C": ["cat", "car", "can"],
        "D": ["dog", "day", "do"],
        "E": ["eat", "egg", "eye"],
        "F": ["fish", "fun", "fly"],
        "G": ["go", "girl", "good"],
        "H": ["hello", "house", "happy"],
        "I": ["I", "is", "in"],
        "J": ["jump", "joy", "jug"],
        "K": ["kite", "king", "kind"],
        "L": ["love", "like", "look"],
        "M": ["my", "man", "mother"],
        "N": ["no", "now", "name"],
        "O": ["okay", "old", "orange"],
        "P": ["play", "pen", "please"],
        "Q": ["queen", "quick", "quiet"],
        "R": ["run", "red", "rain"],
        "S": ["sun", "see", "sit"],
        "T": ["the", "this", "time"],
        "U": ["you", "up", "under"],
        "V": ["very", "van", "voice"],
        "W": ["we", "what", "where"],
        "X": ["xylophone", "x-ray", "box"],
        "Y": ["you", "your", "yellow"],
        "Z": ["zoo", "zero", "zebra"]
    }
    return word_dict.get(alphabet, [])

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        print("Invalid request: No image data found")
        return jsonify({"error": "Invalid request"}), 400

    try:
        # Decode the base64 image
        image_data = data['image']
        print("Received image data")
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Failed to decode image")
            return jsonify({"error": "Failed to decode image"}), 400

        print("Image decoded successfully")

        # Initialize hand detector
        detector = HandDetector(maxHands=2)

        # Detect hands in the image
        hands, _ = detector.findHands(img)

        if hands:
            print(f"Hands detected: {len(hands)}")
            # Process the image and make predictions
            img_input = preprocess_image(img)
            predictions = model.predict(np.array([img_input]))
            predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])[0]
            suggestions = get_word_suggestions(predicted_class)

            print(f"Prediction: {predicted_class}, Suggestions: {suggestions}")

            # Return the results
            result = {
                "prediction": predicted_class,
                "suggestions": suggestions,
                "selected_words": []  # You can update this based on user input
            }
            return jsonify(result)
        else:
            print("No hands detected")
            return jsonify({"error": "No hands detected"}), 400

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": "Failed to process image"}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)