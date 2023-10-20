import cv2
import numpy as np

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 for the default camera, you may need to adjust this

# Initialize the background frame
background = None

# Set a threshold for motion detection
threshold = 8000

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background is None:
        # Set the initial background frame
        background = gray
        continue

    # Calculate the absolute difference between the current frame and background
    frame_delta = cv2.absdiff(background, gray)

    # Apply a threshold to the frame delta
    _, thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < threshold:
            continue

        # If motion is detected, you can add actions here (e.g., sending alerts)
        # For demonstration, we'll draw a bounding box around the moving object
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Security Camera", frame)

    # If you press 'q', the loop will exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
