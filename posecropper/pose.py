"""MediaPipe Pose wrapper and the pose-landmark overlay (the production diagnostic
reused as a visualization layer). Core deps only (mediapipe, opencv, numpy)."""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def load_pose_detector():
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)


def calculate_body_center(landmarks, image_width, image_height):
    """Body anchor = midpoint of the landmark bounding box (full-skeleton extent)."""
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]
    center_x = int((min(xs) + max(xs)) / 2)
    center_y = int((min(ys) + max(ys)) / 2)
    return center_x, center_y

# *** ALTERNATE HIP-BASED CENTER (considered, left for documentation) ***
# left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
# right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
# center_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
# center_y = int(((left_hip.y + right_hip.y) / 2) * image_height)


def detect(pose_detector, image_rgb):
    """Return (pose_landmarks, landmarks_list, center_x, center_y)."""
    h, w = image_rgb.shape[:2]
    results = pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        raise ValueError("No landmarks detected in the image")
    landmarks_list = results.pose_landmarks.landmark
    cx, cy = calculate_body_center(landmarks_list, w, h)
    return results.pose_landmarks, landmarks_list, cx, cy


def annotate_overlay(image_rgb, pose_landmarks, center_x, center_y):
    """All 33 landmarks + skeleton + labeled computed center + per-landmark names."""
    out = image_rgb.copy()
    h, w, _ = out.shape
    mp_drawing.draw_landmarks(
        out, pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
    )
    cv2.circle(out, (center_x, center_y), 10, (255, 0, 0), -1)
    cv2.putText(out, "CALCULATED CENTER", (center_x + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    for idx, lm in enumerate(pose_landmarks.landmark):
        lx, ly = int(lm.x * w), int(lm.y * h)
        cv2.putText(out, mp_pose.PoseLandmark(idx).name, (lx + 5, ly - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return out
