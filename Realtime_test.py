import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model("AtoZsign_language_model.h5")

# Define the list of classes (in the same order as during training)
classes = ["A", "B", "C","D","E","F","G", "H", "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W" ,"X", "Y", "Z"]

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 350  # Fixed size for the output images

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input.
    """
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image / 255.0  # Normalize pixel values
    return image

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detect hands

    if hands:
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

                # Display the predicted class on the image
                cv2.putText(img, f"Prediction: {predicted_class}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the white background image
                cv2.imshow("Combined Hands", imgWhite)

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

                # Display the predicted class on the image
                cv2.putText(img, f"Prediction: {predicted_class}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the white background image
                cv2.imshow("Single Hand", imgWhite)

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()