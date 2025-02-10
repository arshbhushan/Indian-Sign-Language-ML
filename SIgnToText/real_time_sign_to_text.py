import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import dlib
from scipy.spatial import distance as dist
import time
# Load the trained model
model = tf.keras.models.load_model("AtoZsign_language_model.h5")

offset = 20
imgSize = 350

# Define the list of classes (in the same order as during training)
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Initialize dlib's face detector and facial landmark predictor
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in the same directory

# Constants for EAR-based blink detection
EYE_AR_THRESH = 0.25  # EAR threshold to consider a blink
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm a blink


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input.
    """
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image / 255.0  # Normalize pixel values
    return image

# Function to get word suggestions
def get_word_suggestions(alphabet):
    """
    Provide word suggestions based on the detected alphabet.
    """
    # Example: Use a predefined dictionary or word list
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

# List to store selected words
selected_words = []

blink_counter = 0
blink_frames = 0
selection_made = False  # Flag to track if a selection has been made
last_blink_time = 0  # Track the time of the last blink

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    # Check if hands are visible
    if hands:
        # If hands are visible, proceed with blink detection
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib
        faces = detector_dlib(gray)
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Extract eye landmarks (indices for left and right eyes)
            left_eye = landmarks[36:42]  # Left eye landmarks
            right_eye = landmarks[42:48]  # Right eye landmarks

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0  # Average EAR

            # Check if EAR is below the threshold (blink detected)
            if ear < EYE_AR_THRESH:
                blink_frames += 1
            else:
                if blink_frames >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                    print(f"Blink Detected! Total Blinks: {blink_counter}")
                    last_blink_time = time.time()  # Update the last blink time
                    # Cap the blink counter at 3
                    if blink_counter > 3:
                        blink_counter = 3
                blink_frames = 0  # Reset blink frames after a blink is confirmed

        # Display blink count on the screen
        cv2.putText(img, f"Blinks: {blink_counter}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If two hands are detected
        if len(hands) == 2:
            # Get bounding boxes for both hands
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

            # Create a white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Check if the cropped image is valid (non-empty)
            if imgCrop.size == 0:
                print("Warning: Cropped image is empty.")
            else:
                # Resize and paste the cropped image onto the white background
                aspect_ratio = (y_max - y_min) / (x_max - x_min)
                if aspect_ratio > 1:
                    # Combined region is taller than wide
                    k = imgSize / (y_max - y_min)
                    w_cal = int(k * (x_max - x_min))
                    img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                    w_gap = (imgSize - w_cal) // 2
                    imgWhite[:, w_gap:w_gap + w_cal] = img_resize
                else:
                    # Combined region is wider than tall
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
                print(f"Predicted Class: {predicted_class}")

                # Get word suggestions for the predicted alphabet
                suggestions = get_word_suggestions(predicted_class)

                # Display the predicted class and suggestions on the image
                cv2.putText(img, f"Prediction: {predicted_class}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for i, word in enumerate(suggestions):
                    cv2.putText(img, f"{i + 1}. {word}", (10, 100 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Use blink counter to select a word (only if both hands are visible)
                if blink_counter > 0 and not selection_made:
                    # Check if 3 seconds have passed since the last blink
                    if time.time() - last_blink_time > 3:
                        if blink_counter == 1:
                            selected_word = predicted_class  # Alphabet itself
                        elif blink_counter == 2:
                            selected_word = suggestions[0]  # First suggestion
                        elif blink_counter == 3:
                            selected_word = suggestions[1]  # Second suggestion
                        else:
                            selected_word = None

                        if selected_word:
                            selected_words.append(selected_word)
                            print(f"Selected Word: {selected_word}")
                            selection_made = True  # Mark selection as made
                            blink_counter = 0  # Reset blink counter after selection

        # If only one hand is detected
        elif len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            # Create a white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if imgCrop.size == 0:
                print("Warning: Cropped image is empty.")
            else:
                # Resize and paste the cropped image onto the white background
                aspect_ratio = h / w
                if aspect_ratio > 1:
                    # Hand is taller than wide
                    k = imgSize / h
                    w_cal = int(k * w)
                    img_resize = cv2.resize(imgCrop, (w_cal, imgSize))
                    w_gap = (imgSize - w_cal) // 2
                    imgWhite[:, w_gap:w_gap + w_cal] = img_resize
                else:
                    # Hand is wider than tall
                    k = imgSize / w
                    h_cal = int(k * h)
                    img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                    h_gap = (imgSize - h_cal) // 2
                    imgWhite[h_gap:h_gap + h_cal, :] = img_resize

                # Preprocess the image for model input
                img_input = preprocess_image(imgWhite)

                # Make a prediction
                predictions = model.predict(np.array([img_input]))
                predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])[0]
                print(f"Predicted Class: {predicted_class}")

                # Get word suggestions for the predicted alphabet
                suggestions = get_word_suggestions(predicted_class)

                # Display the predicted class and suggestions on the image
                cv2.putText(img, f"Prediction: {predicted_class}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for i, word in enumerate(suggestions):
                    cv2.putText(img, f"{i + 1}. {word}", (10, 100 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Allow the user to select a word
                key = cv2.waitKey(1)
                if key in [ord(str(i + 1)) for i in range(len(suggestions))]:
                    selected_word = suggestions[key - ord('1')]
                    selected_words.append(selected_word)
                    print(f"Selected Word: {selected_word}")
                elif key == ord(' '):  # Space bar to add a space
                    selected_words.append(" ")  # Add a space
                    print("Space added")
                elif key == ord('r'):  # Reset or erase the last word
                    if selected_words:
                        selected_words.pop()
                        print("Last word erased")

    # If no hands are visible, reset blink counter and selection flag
    else:
        blink_counter = 0
        blink_frames = 0
        selection_made = False  # Reset selection flag

    # Display the formed sentence on the screen
    sentence = "".join(selected_words)  # Join words without spaces
    cv2.putText(img, f"Sentence: {sentence}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()