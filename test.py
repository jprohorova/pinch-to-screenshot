import os
import time
import subprocess
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load MediaPipe hand model
base_option = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

save_dir = "screenshots"
os.makedirs(save_dir, exist_ok=True)

last_shot_time = 0
cooldown = 2.0

def take_screenshot():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")
    subprocess.run(["screencapture", filename])
    print(f"Screenshot saved: {filename}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]

                sx, sy = int(thumb_tip.x * w), int(thumb_tip.y * h)
                dx, dy = int(index_tip.x * w), int(index_tip.y * h)

                cv2.circle(frame, (sx, sy), 10, (0, 0, 0), -1)
                cv2.circle(frame, (dx, dy), 10, (0, 0, 0), -1)
                cv2.line(frame, (sx, sy), (dx, dy), (255, 0, 0), 1)

                dist = ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5

                cv2.putText(
                    frame, 
                    f"Distance: {int(dist)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1
                )

                current_time = time.time()
                if dist < 40 and (current_time - last_shot_time) > cooldown:
                    take_screenshot()
                    last_shot_time = current_time

        cv2.imshow("Pinch to screenshot", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()