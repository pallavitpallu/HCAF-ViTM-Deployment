# HCAF-ViT-M: Hybrid CNN and Vision Transformer for Driver Drowsiness Detection

## Project Overview

HCAF-ViT-M is a software-based driver drowsiness detection system designed to identify signs of driver fatigue using facial features and deep learning.

The system combines Convolutional Neural Networks (CNN), Vision Transformers (ViT), and behavioural feature analysis to classify driver states as Active or Fatigue.

## Key Features

- Real-time webcam-based face detection.
- Facial landmark extraction using MediaPipe.
- Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) analysis.
- Head movement and behavioural feature extraction.
- Hybrid CNN–Vision Transformer architecture.
- Real-time fatigue prediction.
- Audible fatigue warning.
- Interactive Streamlit dashboard.
- Model performance visualisation.

## Proposed Architecture

The proposed HCAF-ViT-M architecture consists of three branches:

1. **CNN branch:** ResNet18 extracts spatial facial features.
2. **Vision Transformer branch:** ViT captures global facial representations.
3. **Behavioural feature branch:** An MLP processes four behavioural features:
   - EAR
   - MAR
   - Horizontal head movement
   - Vertical head movement

The extracted representations are combined through a fusion network for final classification.

## Dataset

**Dataset:** UTA-RLDD

The image dataset used in this implementation contains:

| Split | Images |
|---|---:|
| Training | 5,539 |
| Validation | 692 |
| Testing | 693 |
| **Total** | **6,924** |

The implemented classifier uses two classes:

- Active
- Fatigue

## Model Performance

Evaluation on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 93.62% |
| Balanced Accuracy | 93.61% |
| Macro F1-score | 93.62% |
| ROC-AUC | 98.11% |
| Matthews Correlation Coefficient | 0.873 |

### Confusion Matrix

| | Predicted Active | Predicted Fatigue |
|---|---:|---:|
| Actual Active | 436 | 20 |
| Actual Fatigue | 38 | 415 |

## Technology Stack

- Python
- PyTorch
- TorchVision
- Hugging Face Transformers
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Streamlit
- Streamlit-WebRTC

## Project Structure

```text
HCAF-ViTM-Deployment/
├── app.py
├── model.py
├── face_detector.py
├── face_features.py
├── scaler.py
├── style.py
├── test_prediction.py
├── test_webrtc.py
├── requirements.txt
├── checkpoints/
│   └── best_model.pt
├── logs/
│   └── behavior_scaler.json
├── models/
│   └── face_landmarker.task
└── pages/
    ├── 1_Project_Overview.py
    └── 2_Model_Performance.py
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd HCAF-ViTM-Deployment
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Allow camera access when prompted by your browser.

## Model Files

The trained model checkpoint is stored separately using Git Large File Storage (Git LFS).

The MediaPipe face landmark model and behavioural scaler are also required for inference.

## Limitations

- The implementation currently supports two classification states: Active and Fatigue.
- Webcam performance depends on lighting, camera quality, and network conditions.
- The system is an academic prototype and is not a certified vehicle safety system.

## Future Enhancements

- Introduce a Low Vigilance classification state.
- Improve temporal fatigue analysis.
- Evaluate performance across different lighting and driving conditions.
- Explore deployment on edge devices.

## References

1. Ghoddoosian, A., Galib, M., and Athitsos, V. (2019). A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection.
2. Dosovitskiy, A. et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.
3. He, K. et al. (2016). Deep Residual Learning for Image Recognition.

---

**Note:** This project is developed for academic and research purposes.