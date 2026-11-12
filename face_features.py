
import numpy as np

# MediaPipe Face Landmarker landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks
MOUTH = [61, 13, 14, 291, 12, 15]


def distance(p1, p2):
    """Euclidean distance between two normalized landmark points."""
    return np.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(landmarks):
    """Calculate Eye Aspect Ratio."""
    left = LEFT_EYE
    right = RIGHT_EYE

    left_ear = (
        distance(landmarks[left[1]], landmarks[left[5]]) +
        distance(landmarks[left[2]], landmarks[left[4]])
    ) / (2.0 * distance(landmarks[left[0]], landmarks[left[3]]))

    right_ear = (
        distance(landmarks[right[1]], landmarks[right[5]]) +
        distance(landmarks[right[2]], landmarks[right[4]])
    ) / (2.0 * distance(landmarks[right[0]], landmarks[right[3]]))

    return float((left_ear + right_ear) / 2.0)


def calculate_mar(landmarks):
    """Calculate Mouth Aspect Ratio."""
    vertical_1 = distance(landmarks[MOUTH[1]], landmarks[MOUTH[5]])
    vertical_2 = distance(landmarks[MOUTH[2]], landmarks[MOUTH[4]])
    horizontal = distance(landmarks[MOUTH[0]], landmarks[MOUTH[3]])

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calculate_head_features(landmarks):
    """Estimate head movement using nose and eye-center positions."""
    nose = landmarks[1]

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    eye_center_x = (left_eye.x + right_eye.x) / 2.0
    eye_center_y = (left_eye.y + right_eye.y) / 2.0

    horizontal_offset = float(nose.x - eye_center_x)
    vertical_offset = float(nose.y - eye_center_y)

    return horizontal_offset, vertical_offset


def extract_frame_features(face_landmarker_result):
    """Extract behavioral features from one MediaPipe result."""

    if not face_landmarker_result.face_landmarks:
        return {
            "face_detected": 0,
            "ear": np.nan,
            "mar": np.nan,
            "head_horizontal": np.nan,
            "head_vertical": np.nan
        }

    landmarks = face_landmarker_result.face_landmarks[0]

    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    head_h, head_v = calculate_head_features(landmarks)

    return {
        "face_detected": 1,
        "ear": ear,
        "mar": mar,
        "head_horizontal": head_h,
        "head_vertical": head_v
    }
