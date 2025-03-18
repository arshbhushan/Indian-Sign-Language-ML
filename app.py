from flask import Flask, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import dlib
from scipy.spatial import distance as dist
import time
import eventlet

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSockets

# Global variables to store predictions, blink counter, and sentence
current_prediction = ""
current_suggestions = []
blink_counter = 0
sentence = ""
blink_frames = 0
last_blink_time = 0
selection_made = False

# Load the model and initialize components
model = tf.keras.models.load_model("AtoZsign_language_model.h5")
offset = 20
imgSize = 350
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

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

def generate_frames():
    global current_prediction, current_suggestions, blink_counter, sentence, blink_frames, last_blink_time, selection_made

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Detect hands
        hands, img = detector.findHands(img)

        if hands:
            print(f"Detected {len(hands)} hands")
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces using dlib
            faces = detector_dlib(gray)
            for face in faces:
                # Get facial landmarks
                landmarks = predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                # Extract eye landmarks
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]

                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Check for blink
                if ear < EYE_AR_THRESH:
                    blink_frames += 1
                else:
                    if blink_frames >= EYE_AR_CONSEC_FRAMES:
                        blink_counter += 1
                        last_blink_time = time.time()
                        if blink_counter > 3:
                            blink_counter = 3

                        # Use blink counter to select a word
                        if blink_counter > 0 and not selection_made:
                            if time.time() - last_blink_time > 3:
                                if blink_counter == 1:
                                    selected_word = current_prediction  # Alphabet itself
                                elif blink_counter == 2:
                                    selected_word = current_suggestions[0]  # First suggestion
                                elif blink_counter == 3:
                                    selected_word = current_suggestions[1]  # Second suggestion
                                else:
                                    selected_word = None

                                if selected_word:
                                    sentence += selected_word + " "  # Add word to the sentence
                                    selection_made = True  # Mark selection as made
                                    blink_counter = 0  # Reset blink counter after selection

                    blink_frames = 0  # Reset blink_frames after a blink is confirmed

            # If two hands are detected
            if len(hands) == 2:
                hand1 = hands[0]
                hand2 = hands[1]
                x1, y1, w1, h1 = hand1['bbox']
                x2, y2, w2, h2 = hand2['bbox']

                # Calculate the combined bounding box
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)

                # Crop the combined region
                imgCrop = img[y_min-offset:y_max+offset, x_min-offset:x_max+offset]

                if imgCrop.size != 0:
                    # Create a white background image
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    # Resize and paste the cropped image onto the white background
                    aspect_ratio = (y_max - y_min) / (x_max - x_min)
                    if aspect_ratio > 1:
                        k = imgSize / (y_max - y_min)
                        w_cal = int(k * (x_max - x_min))
                        img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                        w_gap = (imgSize - w_cal) // 2
                        imgWhite[:, w_gap:w_gap + w_cal] = img_resize
                    else:
                        k = imgSize / (x_max - x_min)
                        h_cal = int(k * (y_max - y_min))
                        img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                        h_gap = (imgSize - h_cal) // 2
                        imgWhite[h_gap:h_gap + h_cal, :] = img_resize

                    # Preprocess the image for model input
                    img_input = preprocess_image(imgWhite)

                    # Make a prediction
                    predictions = model.predict(np.array([img_input]))
                    predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])[0]
                    print(f"Predicted class: {predicted_class}")
                    current_prediction = predicted_class
                    current_suggestions = get_word_suggestions(predicted_class)                 

        # If no hands are visible, reset blink counter and selection flag
        else:
            blink_counter = 0
            blink_frames = 0
            selection_made = False  # Reset selection flag

        # Send updates to the client via WebSocket
        print(f"Emitting update: {current_prediction}, {current_suggestions}, {blink_counter}, {sentence}")  # Log the emitted data
        socketio.emit('update', {
            "prediction": current_prediction,
            "suggestions": current_suggestions,
            "blink_counter": blink_counter,
            "sentence": sentence
        })

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Add the /video_feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)