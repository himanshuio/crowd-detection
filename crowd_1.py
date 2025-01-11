#pip install ultralytics numpy supervision opencv-python-headless
#sudo apt-get install python3-tk
import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
import tkinter as tk
from tkinter import simpledialog
import RPi.GPIO as GPIO
import time

# Path to YOLO model
MODEL_PATH_CROWD = "D:/43_himanshu_dhomane_/yolov8n.pt"  # For crowd detection

# GPIO setup for Raspberry Pi
BUZZER_PIN = 18  # Replace with your GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Global variables
people_count = 0
threshold = 0

# Function to activate the buzzer
def activate_buzzer():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# Function to process and display video with crowd detection
def process_video(video_path, model_path, threshold):
    global people_count

    print(f"Trying to open video: {video_path}")

    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file or camera
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame from video")
        return

    height, width, channels = frame.shape

    # Define the region of interest (ROI) for crowd detection
    polygons = np.array([
        [0, 0],
        [width - 5, 0],
        [width - 5, height - 5],
        [0, height - 5]
    ])

    zones = sv.PolygonZone(polygon=polygons, frame_resolution_wh=(width, height))
    box_annotators = sv.BoxAnnotator(thickness=1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        # Run YOLO detection on the current frame
        results = model(frame, imgsz=1248)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Apply the ROI mask and filter detections
        mask = zones.trigger(detections=detections)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5) & mask]

        # Update the global count of detected people
        people_count = len(detections)

        # Check if the count exceeds the threshold
        if people_count > threshold:
            print(f"Threshold exceeded! Detected: {people_count}, Threshold: {threshold}")
            activate_buzzer()

        # Annotate the frame with detection results
        frame = box_annotators.annotate(scene=frame, detections=detections)

        # Annotate each detected person
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Add the total people count to the frame
        text = f"People detected: {people_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 1)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 10
        cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 0), 1)

        # Display the processed frame
        cv2.imshow("Crowd Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video processing stopped by user.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

# Tkinter GUI for threshold input
def get_threshold_and_start():
    global threshold
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Prompt the user for the threshold
    threshold = simpledialog.askinteger("Input", "Enter the crowd threshold:", minvalue=1, maxvalue=100)

    if threshold:
        print(f"Threshold set to: {threshold}")
        video_path = "D:/project/vid/crowd.mp4"  # Replace with your video path or use '0' for webcam
        process_video(video_path, MODEL_PATH_CROWD, threshold)

if __name__ == "__main__":
    try:
        get_threshold_and_start()
    finally:
        GPIO.cleanup()  # Clean up GPIO pins on exit
