import cv2
import numpy as np

def detect_anomaly(frame, prev_frame):
    """
    Enhanced anomaly detection logic:
    1. Brightness: Flags frames with high brightness as anomalous.
    2. Motion: Detects sudden changes in motion as anomalies.
    3. Unusual shapes: Identifies irregular contours as anomalies.
    """
    anomalies = []

    # 1. Brightness-based anomaly detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness > 200:  # Example brightness threshold
        anomalies.append("High Brightness")

    # 2. Motion-based anomaly detection
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = np.sum(thresh) / 255  # Calculate motion intensity
        if motion > 10000:  # Example motion threshold
            anomalies.append("Sudden Motion")

    # 3. Shape-based anomaly detection
    edges = cv2.Canny(gray, 50, 150)  # Detect edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Example area threshold for unusual shapes
            anomalies.append("Irregular Shape")

    return anomalies


def run_anomaly_detection():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        anomalies = detect_anomaly(frame, prev_frame)

        # Update the previous frame
        prev_frame = gray_frame

        # Label based on detected anomalies
        if anomalies:
            label = f"Anomaly Detected: {', '.join(anomalies)}"
            color = (0, 0, 255)  # Red for anomalies
        else:
            label = "Normal"
            color = (0, 255, 0)  # Green for normal

        # Display the label on the frame
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Anomaly Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_anomaly_detection()
