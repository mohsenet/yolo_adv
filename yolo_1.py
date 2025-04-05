import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Nano model (small & fast)

# Define door region (x1, y1, x2, y2)
door_roi = (300, 200, 600, 600)  # Adjust based on your camera view

# Variables for tracking
person_in_roi = False
entry_time = None
duration_threshold = 10  # Alert if person stays >10 sec

# Open CCTV feed (use RTSP/HTTP or local camera)
cap = cv2.VideoCapture("rtsp://your_cctv_feed")  # Replace with your feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people using YOLO
    results = model(frame, classes=0)  # Class 0 = person in COCO dataset

    # Check if any person is in the door ROI
    person_detected = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Check if person is inside door ROI
        if (x1 > door_roi[0] and y1 > door_roi[1] and x2 < door_roi[2] and y2 < door_roi[3]):
            person_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

    # Track duration
    if person_detected:
        if not person_in_roi:
            person_in_roi = True
            entry_time = time.time()
        else:
            duration = time.time() - entry_time
            if duration > duration_threshold:
                print(f"Alert! Person at door for {duration:.1f} sec")
                cv2.putText(frame, f"ALERT: {duration:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        person_in_roi = False

    # Draw door ROI
    cv2.rectangle(frame, (door_roi[0], door_roi[1]), (door_roi[2], door_roi[3]), (255, 0, 0), 2)

    # Display
    cv2.imshow("CCTV Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()