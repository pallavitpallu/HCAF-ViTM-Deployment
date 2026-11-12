
import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import HCAFViTM
from face_detector import create_face_landmarker, detect_face
from scaler import scale_features


# 1. Load the saved model
device = torch.device("cpu")

model = HCAFViTM()
checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()

print("HCAF-ViT-M checkpoint loaded successfully.")


# 2. Prepare image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 3. Capture one webcam frame
cap = cv2.VideoCapture(0)
landmarker = create_face_landmarker()

try:
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Could not capture a webcam frame.")

    # 4. Extract facial and behavioral features
    features, result = detect_face(frame, landmarker)

    print("Face detected:", features["face_detected"])

    if not features["face_detected"]:
        print("No face detected. Prediction skipped.")

    else:
        # 5. Prepare image input
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image_tensor = transform(pil_image).unsqueeze(0)

        # 6. Scale behavioral features
        behavioral_values = scale_features(
            features["ear"],
            features["mar"],
            features["head_horizontal"],
            features["head_vertical"]
        )

        behavioral_tensor = torch.tensor(
            behavioral_values,
            dtype=torch.float32
        )

        # 7. Run model prediction
        with torch.no_grad():
            logits = model(image_tensor, behavioral_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = int(probabilities.argmax())
        class_names = ["Active", "Fatigue"]

        print("\n----- PREDICTION RESULT -----")
        print("Prediction:", class_names[predicted_class])
        print("Active probability:", round(probabilities[0].item() * 100, 2), "%")
        print("Fatigue probability:", round(probabilities[1].item() * 100, 2), "%")

finally:
    landmarker.close()
    cap.release()
