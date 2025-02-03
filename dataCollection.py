import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300  # Fixed size for the output images

folder = "Data/A"

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

            # Crop images for both hands
            imgCrop1 = img[y1-offset:y1+h1+offset, x1-offset:x1+w1+offset]
            imgCrop2 = img[y2-offset:y2+h2+offset, x2-offset:x2+w2+offset]

            # Create white background images for both hands
            imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgWhite2 = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Check if both cropped images are valid (non-empty)
            if imgCrop1.size == 0 or imgCrop2.size == 0:
                print("Warning: One of the cropped images is empty. Skipping concatenation.")
            else:
                # Resize and paste the cropped images onto the white background
                aspect_ratio1 = h1 / w1
                if aspect_ratio1 > 1:
                    # Hand is taller than wide
                    k = imgSize / h1
                    w_cal = int(k * w1)
                    img_resize1 = cv2.resize(imgCrop1, (w_cal, imgSize))
                    w_gap = (imgSize - w_cal) // 2
                    imgWhite1[:, w_gap:w_gap + w_cal] = img_resize1
                else:
                    # Hand is wider than tall
                    k = imgSize / w1
                    h_cal = int(k * h1)
                    img_resize1 = cv2.resize(imgCrop1, (imgSize, h_cal))
                    h_gap = (imgSize - h_cal) // 2
                    imgWhite1[h_gap:h_gap + h_cal, :] = img_resize1

                aspect_ratio2 = h2 / w2
                if aspect_ratio2 > 1:
                    # Hand is taller than wide
                    k = imgSize / h2
                    w_cal = int(k * w2)
                    img_resize2 = cv2.resize(imgCrop2, (w_cal, imgSize))
                    w_gap = (imgSize - w_cal) // 2
                    imgWhite2[:, w_gap:w_gap + w_cal] = img_resize2
                else:
                    # Hand is wider than tall
                    k = imgSize / w2
                    h_cal = int(k * h2)
                    img_resize2 = cv2.resize(imgCrop2, (imgSize, h_cal))
                    h_gap = (imgSize - h_cal) // 2
                    imgWhite2[h_gap:h_gap + h_cal, :] = img_resize2

                # Combine the white background images horizontally
                combined_img = cv2.hconcat([imgWhite1, imgWhite2])

                # Display the combined image
                cv2.imshow("Combined Hands", combined_img)

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

                # Display the white background image
                cv2.imshow("Single Hand", imgWhite)

    # Display the original image with hand landmarks
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()