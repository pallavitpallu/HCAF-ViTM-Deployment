import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_features import extract_frame_features

MODEL_PATH = Path("models/face_landmarker.task")


def create_face_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face model not found: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


def detect_face(frame, landmarker):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    features = extract_frame_features(result)

    return features, result