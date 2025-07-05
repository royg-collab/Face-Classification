
# Task_B: Multi-Model Face Recognition Pipeline (Clean + Distorted)

## Project Objective

This project tackles a **multi-class face recognition** challenge under **real-world conditions**, including **distortions** like blur, low light, and occlusion.

The goal is to build a **modular AI system** that can:
- Detect whether an input face image is distorted
- Route it to the appropriate face recognition model
- Predict the correct person identity

---

##  Dataset Structure

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

##  Approach Overview

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

##  Model Summary

| Model ID | Purpose                   | Architecture             | Training Data |
|----------|---------------------------|---------------------------|---------------|
| Model 1  | Distortion Classifier      | Custom CNN                | Clean + Distorted |
| Model 2  | Face Recognition (Clean)   | MobileNetV3 Small + SE    | Clean images only |
| Model 3  | Face Recognition (Distorted)| MobileNetV3 Small + SE   | Distorted images only |

All models are trained separately and saved, then unified in a prediction pipeline.

---

## UnifiedFaceAnalyzer

A 3-stage deep learning pipeline for face recognition across both clean and distorted face images using MobileNetV3 in PyTorch.


- This project combines three specialized models:
- Distortion Classifier – Identifies whether an input face is clean or distorted.
- Clean Face Recognizer – Recognizes identities from clean face images.
- Distorted Face Recognizer – Recognizes identities from distorted face images (e.g., foggy, blurry, rainy).
These models are unified into a single class: UnifiedFaceAnalyzer.

✅ The .pt file (UnifiedFaceAnalyzer.pt) stores the model weights only. To use it, the architecture must be defined in unified_face_analyzer.py.

✅ Commit both .pt and .py files to GitHub for full reproducibility.

##  Evaluation Results
###  Classification Report + Confusion Matrix
```
Classification Report:
              precision    recall  f1-score   support

 001_frontal       1.00      1.00      1.00         3
 002_frontal       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```
-  `confusion_matrix.png`
![image](https://github.com/user-attachments/assets/818bfd73-792e-49b2-99bb-dc5cd59ad9d5)

---

##  Saved Model Pipeline

`multi_model_face_recognizer.pt` includes:
- Model 1: Distortion detector
- Model 2: Clean recognizer
- Model 3: Distorted recognizer
- Auto-routing logic

---

##  Setup Instructions

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

##  Repository Structure
```
| File                             | Description                              |
|----------------------------------|------------------------------------------|
├── main.py                            # Combine models into one pipeline     |
├── test.py                            # Evaluate predictions on test images  |
├── unified_face_analyzer.py          # Core unified 3-stage model class      |
├── UnifiedFaceAnalyzer.pt            # Saved unified model weights (optional)|
├── model1_distortion_classifier.pth  # Clean vs Distorted classifier         |
├── clean_mobilenetv3_face_recognition.pth  # Clean face recognizer           |
├── mobilenetv3_model_distorted_877.pth     # Distorted face recognizer       |
├── class_to_idx_clean.json           # Label map for clean images            |
├── class_to_idx_877.json             # Label map for distorted images        |
└── test_images/                      # Folder of test images                 |
```
---
## Testing

A separate test.py script is provided that:
*Accepts a test dataset path (flat folder of images)
*Loads the pretrained unified model
*Outputs classification report and confusion matrix
```
python test.py \
    --model_path UnifiedFaceAnalyzer.pt \
    --test_path /path/to/test_images
```
---
## Test Image Format
Your test_images/ folder should contain files named like:
```
001_frontal_clear.jpg
001_frontal_foggy.jpg
002_frontal_blur.jpg
001_frontal_rainy.jpg
```
*Each file is matched to a label in true_labels inside test.py
---
---
## File Usage Guide
```
| File                                     | Purpose                                                            | How to Use                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `UnifiedFaceAnalyzer.pt`                 | ✅ Full saved weights of the unified 3-model pipeline               | Load using `torch.load()` **after** importing the `UnifiedFaceAnalyzer` class from `unified_face_analyzer.py` |
| `unified_face_analyzer.py`               | 🧠 Contains the full model architecture                            | Import the `UnifiedFaceAnalyzer` class: `from unified_face_analyzer import UnifiedFaceAnalyzer`               |
| `test.py`                                | 🧪 Script to evaluate images using the full pipeline               | Run `python test.py` to copy test images and print classification report                                      |
| `main.py`                                | 🛠 (Optional) Builds and optionally saves a unified model instance | Can be used to combine and save `.pt` model                                                                   |
| `class_to_idx_clean.json`                | 🔠 Class-to-index mapping for clean faces                          | Required by `UnifiedFaceAnalyzer` to decode predictions                                                       |
| `class_to_idx_877.json`                  | 🔠 Class-to-index mapping for distorted faces                      | Required by `UnifiedFaceAnalyzer` to decode predictions                                                       |
| `mobilenetv3_model_distorted_877.pth`    | 🎭 Distorted face recognizer weights                               | Used internally by `UnifiedFaceAnalyzer`                                                                      |
| `clean_mobilenetv3_face_recognition.pth` | 😊 Clean face recognizer weights                                   | Used internally by `UnifiedFaceAnalyzer`                                                                      |
| `model1_distortion_classifier.pth`       | 🧪 Distortion (clean vs. distorted) classifier weights             | Used internally by `UnifiedFaceAnalyzer`
```
---

##  Key Learnings

- Modular pipelines increase flexibility and robustness
- Distortion-aware models perform better than a single mixed model
- MobileNetV3 (small + SE) works well for fast, light face recognition

---

##  Author
**Gargi Roy** .

---
