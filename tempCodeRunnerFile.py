import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os
import time  # For timing functionality
from tensorflow import keras

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
model = keras.models.load_model("Model/keras_model.h5", compile=False)
model.save("Model/keras_model_fixed.h5")
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 350  # Fixed size for the output images

# Folder to save images
folder = "Data/C"
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0  # Counter to keep track of saved images
save_images = False  # Flag to control automatic saving
start_time = None  # To track when saving starts

labels = ["A","B","C"]

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

                # Display the white background image
                cv2.imshow("Combined Hands", imgWhite)

                # Save the combined image automatically if save_images is True
                if save_images:
                    counter += 1
                    save_path = os.path.join(folder, f"combined_{counter}.jpg")
                    cv2.imwrite(save_path, imgWhite)
                    print(f"Saved combined image to {save_path}")

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
                    prediction, index= classifier.getPrediction(img)
                    print(f"Predicted hand gesture: {prediction} with index {index}")
                else:
                    # Hand is wider than tall
                    k = imgSize / w
                    h_cal = int(k * h)
                    img_resize = cv2.resize(imgCrop, (imgSize, h_cal))
                    h_gap = (imgSize - h_cal) // 2
                    imgWhite[h_gap:h_gap + h_cal, :] = img_resize

                # Display the white background image
                cv2.imshow("Single Hand", imgWhite)

                # Save the single hand image automatically if save_images is True
                if save_images:
                    counter += 1
                    save_path = os.path.join(folder, f"single_{counter}.jpg")
                    cv2.imwrite(save_path, imgWhite)
                    print(f"Saved single hand image to {save_path}")

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)

    # Check for key presses
    cv2.waitKey(1)
