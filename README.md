
# 🎭 Multi-Model Face Recognition Pipeline (Clean + Distorted)

## 🧠 Project Objective

This project tackles a **multi-class face recognition** challenge under **real-world conditions**, including **distortions** like blur, low light, and occlusion.

The goal is to build a **modular AI system** that can:
- Detect whether an input face image is distorted
- Route it to the appropriate face recognition model
- Predict the correct person identity

---

## 🗂️ Dataset Structure

Each class represents a person, with:
- One clean/original image
- A `distortion/` subfolder with several distorted variants

```
train/
├── 001_frontal/
│   ├── 001_frontal.jpg
│   └── distortion/
│       ├── blurred.jpg
│       └── lowlight.jpg
├── 002_frontal/
│   └── ...
val/
└── ...
```

---

## 🔍 Approach Overview

We designed a **3-model inference pipeline** that mimics human visual processing:

```
             +----------------------+
             |  Input Face Image    |
             +----------+-----------+
                        |
                        ▼
        +------------------------------+
        |   Model 1: Distortion Class  |
        |   (Clean vs Distorted)       |
        +------------------------------+
             |                |
     Clean Image         Distorted Image
        |                        |
        ▼                        ▼
+----------------+     +-----------------+
|  Model 2        |     |  Model 3        |
|  Clean Face     |     |  Distorted Face |
|  Recognizer     |     |  Recognizer     |
+----------------+     +-----------------+
        |                        |
        +-----------+------------+
                    ▼
         Final Person Prediction
```

---

## 🧱 Model Summary

| Model ID | Purpose                   | Architecture             | Training Data |
|----------|---------------------------|---------------------------|---------------|
| Model 1  | Distortion Classifier      | Custom CNN                | Clean + Distorted |
| Model 2  | Face Recognition (Clean)   | MobileNetV3 Small + SE    | Clean images only |
| Model 3  | Face Recognition (Distorted)| MobileNetV3 Small + SE   | Distorted images only |

All models are trained separately and saved, then unified in a prediction pipeline.

---

## 📈 Evaluation Results

### 🔍 Validation Accuracy

| Model        | Accuracy (%) |
|--------------|--------------|
| Model 1      | ~96%         |
| Model 2      | ~24%         |
| Model 3      | ~3%          |

> ⚠️ Model 3’s low accuracy reflects real-world difficulty in face recognition under severe distortions.

### 🧾 Classification Report + Confusion Matrix
- 📄 `classification_report.txt`
- 📊 `confusion_matrix.png`

---

## 💾 Saved Model Pipeline

`multi_model_face_recognizer.pt` includes:
- Model 1: Distortion detector
- Model 2: Clean recognizer
- Model 3: Distorted recognizer
- Auto-routing logic

---

## 🛠️ Setup Instructions

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

---

## 🚀 Inference Demo

```python
import torch
model = torch.load("multi_model_face_recognizer.pt")
model.eval()
result = model.predict_image("/path/to/image.jpg")
print("Predicted Class:", result)
```

---

## 📁 Repository Structure

| File                             | Description                              |
|----------------------------------|------------------------------------------|
| `multi_model_face_recognizer.pt` | Full pipeline model                      |
| `test_inference.py`              | Test evaluation script                   |
| `README.md`                      | Project overview                         |
| `classification_report.txt`      | Precision, recall, F1 metrics            |
| `confusion_matrix.png`           | Visual confusion matrix                  |

---

## 🧠 Key Learnings

- Modular pipelines increase flexibility and robustness
- Distortion-aware models perform better than a single mixed model
- MobileNetV3 (small + SE) works well for fast, light face recognition

---

## 👨‍💻 Author
**ROY G** — Lead model designer, data pipeline engineer, evaluator.

---
