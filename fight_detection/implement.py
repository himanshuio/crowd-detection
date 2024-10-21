import cv2
from ultralytics import YOLO
import os

# Define the YOLOv8s model and video file paths
model_path = "D:/project/fight_detection/runs/detect/train/weights/best.pt"  # Using the pre-trained YOLOv8 small model
video_path = "D:/project/fight_detection/vdo/fi055.mp4"
# Check if the model path exists
if not os.path.exists(model_path):
    print(f"Model file does not exist: {model_path}")
else:
    # Load the pre-trained YOLOv8 model
    model = YOLO(model_path)

    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
    else:
        # Open the video using OpenCV
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
        else:
            # Loop through the frames of the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Video stream ended or there was an error.")
                    break

                # Run YOLOv8 model on the current frame
                results = model(frame)

                # Visualize the detections
                annotated_frame = results[0].plot()

                # Display the frame
                cv2.imshow("YOLOv8 Detection", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            # Release the video capture object and close OpenCV windows
            cap.release()
            cv2.destroyAllWindows()
