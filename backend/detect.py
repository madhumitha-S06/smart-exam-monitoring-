from ultralytics import YOLO
import cv2
import csv
import os
model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.80

restricted_objects = [
    "cell phone",
    "scissors",
    "knife"
]

os.makedirs("backend/results", exist_ok=True)
csv_path = "backend/results/detections.csv"

csv_file = open(csv_path, "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "Object",
    "Confidence"
])
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

print("Webcam Started")
print("Press Q to Exit")
while True:

    ret, frame = cap.read()

    if not ret:
        break
        results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )
    results = model(
        frame,
        conf=0.70,
        verbose=False
    )
    annotated_frame = results[0].plot()

    person_count = 0
    alert_detected = False
    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        object_name = model.names[cls]
        writer.writerow([
            object_name,
            round(conf, 2)
        ])
        if object_name == "person":
            person_count += 1
        if object_name in restricted_objects:
            alert_detected = True
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    if alert_detected:
        cv2.putText(
            annotated_frame,
            "ALERT: Restricted Object Detected!",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    cv2.imwrite(
        "backend/results/output.jpg",
        annotated_frame
    )
    cv2.imshow(
        "Smart Human & Object Detection",
        annotated_frame
    )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("Detection Completed")
print("Results saved in backend/results/")