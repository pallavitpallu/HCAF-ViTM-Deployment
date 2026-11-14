
import cv2
import torch
import streamlit as st
import numpy as np
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from model import HCAFViTM
from face_detector import create_face_landmarker, detect_face
from scaler import scale_features
from style import apply_custom_style
from streamlit_autorefresh import st_autorefresh
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HCAF-ViT-M | Driver Monitoring",
    page_icon="🚗",
    layout="wide"
)

apply_custom_style()

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"

DEVICE = torch.device("cpu")

FATIGUE_THRESHOLD = 0.45
CONSECUTIVE_FRAMES = 2

# Run model inference at most about 4 times per second.
INFERENCE_INTERVAL = 0.25


# --------------------------------------------------
# CAMERA FONT
# --------------------------------------------------

FONT_PATH = Path("C:/Windows/Fonts/segoescb.ttf")
FALLBACK_FONT_PATH = Path("C:/Windows/Fonts/arialbd.ttf")


def get_font(size):
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        try:
            return ImageFont.truetype(
                str(FALLBACK_FONT_PATH), size
            )
        except OSError:
            return ImageFont.load_default()


def draw_camera_text(image, lines):
    """
    Draw small bold cursive text directly on the camera.
    No background panel or boxes are used.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    y = 18

    for index, (text, color) in enumerate(lines):

        font_size = 16 if index == 0 else 13
        font = get_font(font_size)

        # Convert OpenCV BGR colour to PIL RGB.
        rgb_color = (color[2], color[1], color[0])

        draw.text(
            (18, y),
            text,
            font=font,
            fill=rgb_color,
            stroke_width=1,
            stroke_fill=(35, 20, 40)
        )

        y += 23

    result = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    image[:] = result


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def download_checkpoint():

    checkpoint_path = hf_hub_download(
        repo_id="Pallavitpallu/HCAF-ViTM-Driver-Drowsiness",
        filename="best_model.pt"
    )

    return checkpoint_path


@st.cache_resource
def load_model():

    model = HCAFViTM()

    checkpoint_path = download_checkpoint()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class HCAFVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()
        self.landmarker = create_face_landmarker()

        self.fatigue_count = 0
        self.last_inference_time = 0.0

        self.last_lines = [
            ("HCAF-ViT-M | LIVE MONITOR",
             (220, 195, 255)),
            ("Waiting for camera frame...",
             (255, 255, 255))
        ]

        self.last_warning = False

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        try:

            # Detect face and extract facial features.
            features, result = detect_face(
                image,
                self.landmarker
            )

            if not features["face_detected"]:

                self.fatigue_count = 0

                self.last_lines = [
                    ("HCAF-ViT-M | LIVE MONITOR",
                     (220, 195, 255)),
                    ("No face detected",
                     (100, 150, 255)),
                    ("Please face the camera",
                     (255, 255, 255))
                ]

                self.last_warning = False

            else:

                current_time = time.monotonic()

                # Run model inference at a controlled interval.
                if (
                    current_time - self.last_inference_time
                    >= INFERENCE_INTERVAL
                ):

                    rgb_image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    pil_image = Image.fromarray(rgb_image)

                    image_tensor = IMAGE_TRANSFORM(
                        pil_image
                    ).unsqueeze(0).to(DEVICE)

                    behavioral_values = scale_features(
                        features["ear"],
                        features["mar"],
                        features["head_horizontal"],
                        features["head_vertical"]
                    )

                    behavioral_tensor = torch.tensor(
                        behavioral_values,
                        dtype=torch.float32
                    ).to(DEVICE)

                    with torch.inference_mode():

                        logits = self.model(
                            image_tensor,
                            behavioral_tensor
                        )

                        probabilities = torch.softmax(
                            logits,
                            dim=1
                        )[0]

                    active_probability = probabilities[0].item()
                    fatigue_probability = probabilities[1].item()

                    prediction = (
                        "Fatigue"
                        if fatigue_probability >= active_probability
                        else "Active"
                    )

                    if fatigue_probability >= FATIGUE_THRESHOLD:
                        self.fatigue_count += 1
                    else:
                        self.fatigue_count = 0

                    self.last_warning = (
                        self.fatigue_count >= CONSECUTIVE_FRAMES
                    )

                    prediction_color = (
                        (0, 0, 255)
                        if self.last_warning
                        else (0, 180, 0)
                    )

                    self.last_lines = [
                        ("HCAF-ViT-M | LIVE MONITOR",
                         (220, 195, 255)),
                        (f"Prediction: {prediction}",
                         prediction_color),
                        (f"Active: {active_probability * 100:.1f}%",
                         (100, 230, 120)),
                        (f"Fatigue: {fatigue_probability * 100:.1f}%",
                         (0, 165, 255)),
                        (f"EAR: {features['ear']:.3f}",
                         (255, 255, 255)),
                        (f"MAR: {features['mar']:.3f}",
                         (255, 255, 255)),
                        (f"Head H: {features['head_horizontal']:.3f}",
                         (220, 200, 240)),
                        (f"Head V: {features['head_vertical']:.3f}",
                         (220, 200, 240))
                    ]

                    self.last_inference_time = current_time

                else:
                    # Refresh feature values between model predictions.
                    # Keep the most recent prediction visible.
                    if len(self.last_lines) >= 8:
                        self.last_lines[4] = (
                            f"EAR: {features['ear']:.3f}",
                            (255, 255, 255)
                        )
                        self.last_lines[5] = (
                            f"MAR: {features['mar']:.3f}",
                            (255, 255, 255)
                        )
                        self.last_lines[6] = (
                            f"Head H: {features['head_horizontal']:.3f}",
                            (220, 200, 240)
                        )
                        self.last_lines[7] = (
                            f"Head V: {features['head_vertical']:.3f}",
                            (220, 200, 240)
                        )

            # Draw the latest available prediction on every frame.
            draw_camera_text(
                image,
                self.last_lines
            )

            if self.last_warning:

                draw_camera_text(
                    image,
                    [
                        ("POSSIBLE FATIGUE DETECTED",
                         (0, 0, 255))
                    ]
                )

        except Exception as error:

            print("Video processing error:", repr(error))

            # Show the processing problem without hiding the video.
            draw_camera_text(
                image,
                [
                    ("CAMERA PROCESSING ERROR",
                     (0, 0, 255))
                ]
            )

        return frame.from_ndarray(
            image,
            format="bgr24"
        )


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.markdown("""
<div class="project-hero">
    <div class="eyebrow">
        INTELLIGENT DRIVER MONITORING SYSTEM
    </div>
    <h1>HCAF-ViT-M</h1>
    <p>
        Hybrid CNN–Vision Transformer based driver drowsiness
        detection using facial and behavioural features.
    </p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.markdown("### Live monitoring")
    st.caption(
        "Real-time facial analysis and fatigue prediction."
    )

with col2:
    st.markdown("### System status")
    st.caption(
        "CPU inference | Camera input | Microphone disabled"
    )


st.info(
    "Allow camera access when your browser asks. "
    "No microphone access is required."
)


# --------------------------------------------------
# CHECKPOINT VALIDATION
# --------------------------------------------------

# --------------------------------------------------
# CHECKPOINT VALIDATION
# --------------------------------------------------

try:

    load_model()

    st.success(
        "Trained HCAF-ViT-M model loaded successfully."
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()

# --------------------------------------------------
# LIVE CAMERA
# --------------------------------------------------

st.markdown("### Driver camera")

st.caption(
    "The model runs on CPU. Predictions update periodically "
    "to help reduce processing load."
)


ctx=webrtc_streamer(
    key="hcaf-vitm-live",
    video_processor_factory=HCAFVideoProcessor,

    rtc_configuration=RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    ),
media_stream_constraints={
    "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15, "max": 20}
    },
    "audio": False
},
    

    async_processing=True
)
# --------------------------------------------------
# FATIGUE ALARM CONTROL
# --------------------------------------------------
# --------------------------------------------------
#-----------------------------------------------
# --------------------------------------------------
# FATIGUE ALARM
# --------------------------------------------------

# --------------------------------------------------
# FATIGUE ALARM
# --------------------------------------------------

import io
import wave
import numpy as np
from streamlit_autorefresh import st_autorefresh

st.markdown("### 🔔 Fatigue alarm")

if "alarm_enabled" not in st.session_state:
    st.session_state.alarm_enabled = False

st.session_state.alarm_enabled = st.toggle(
    "Enable audible fatigue alarm",
    value=st.session_state.alarm_enabled,
    key="fatigue_alarm_toggle"
)

# Read live warning from WebRTC processor
fatigue_alarm_active = False

if ctx.video_processor is not None:
    fatigue_alarm_active = ctx.video_processor.last_warning

# Refresh the warning status every second
st_autorefresh(
    interval=1000,
    key="fatigue_alarm_refresh"
)

alarm_on = (
    st.session_state.alarm_enabled
    and fatigue_alarm_active
)

# Create a short beep
# Create a 3-second repeating fatigue alarm
sample_rate = 44100
duration = 3.0
frequency = 880

t = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    endpoint=False
)

# Repeat beep pulses every 0.8 seconds
pulse_period = 0.8
pulse_duration = 0.45

pulse_position = t % pulse_period
active_pulse = pulse_position < pulse_duration

envelope = np.zeros_like(t)

envelope[active_pulse] = np.sin(
    np.pi * pulse_position[active_pulse] / pulse_duration
)

audio_data = (
    0.30
    * np.sin(2 * np.pi * frequency * t)
    * envelope
)

audio_data = (audio_data * 32767).astype(np.int16)

audio_buffer = io.BytesIO()

with wave.open(audio_buffer, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

if alarm_on:
    st.audio(
        audio_buffer.getvalue(),
        format="audio/wav",
        autoplay=True
    )
    st.error(
        "⚠️ POSSIBLE FATIGUE DETECTED — PLEASE TAKE A BREAK."
    )

elif st.session_state.alarm_enabled:
    st.success("Audible alarm enabled. Monitoring fatigue.")

else:
    st.info("Audible alarm is disabled.")
st.divider()

st.markdown("### Detection configuration")

config1, config2 = st.columns(2)

with config1:

    st.metric(
        "Fatigue probability threshold",
        f"{FATIGUE_THRESHOLD:.2f}"
    )

with config2:

    st.metric(
        "Consecutive frames required",
        CONSECUTIVE_FRAMES
    )


st.warning(
    "Research prototype only — this system is not a certified "
    "driver-safety device. Predictions may be incorrect and "
    "must not be used as the sole measure of driver fitness."
)
