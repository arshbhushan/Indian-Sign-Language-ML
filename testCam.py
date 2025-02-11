import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible")
else:
    print("Webcam is accessible")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame from the webcam")
    else:
        print("Failed to read a frame from the webcam")

cap.release()