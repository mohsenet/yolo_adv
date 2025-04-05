import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model (small and fast for live processing)
model = YOLO("yolov8n.pt")  # or "yolov8s.pt" for better accuracy

# Define your door's ROI (Region of Interest)
door_roi = (300, 200, 600, 600)  # (x1, y1, x2, y2) - Adjust to your camera view

# Variables for tracking
person_in_roi = False
entry_time = None
duration_threshold = 10  # Alert after 10 seconds

# Open the CCTV stream (replace with your RTSP/HTTP URL)
cctv_url = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(cctv_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Check stream URL.")
        break

    # Detect people in the frame
    results = model(frame, classes=0, verbose=False)  # Class 0 = person

    # Check if any detected person is inside the door ROI
    person_detected = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Check if person is inside the door ROI
        if (x1 > door_roi[0] and y1 > door_roi[1] and x2 < door_roi[2] and y2 < door_roi[3]):
            person_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            break

    # Track how long the person stays
    if person_detected:
        if not person_in_roi:  # Person just entered
            person_in_roi = True
            entry_time = time.time()
        else:  # Person is still in ROI
            duration = time.time() - entry_time
            if duration > duration_threshold:
                alert_text = f"ALERT: Person at door for {duration:.1f}s"
                cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(alert_text)  # Log to console
    else:
        person_in_roi = False  # Person left

    # Draw the door ROI
    cv2.rectangle(frame, (door_roi[:2]), (door_roi[2:]), (255, 0, 0), 2)

    # Display the live feed
    cv2.imshow("Live CCTV Monitoring", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()