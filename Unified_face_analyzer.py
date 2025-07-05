#unified_face_analyzer
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class UnifiedFaceAnalyzer(nn.Module):
    def __init__(self,
                 model1_path,
                 model2_path,
                 model3_path,
                 class_map_clean='class_to_idx_clean.json',
                 class_map_distorted='class_to_idx_distorted_877.json'):
        super(UnifiedFaceAnalyzer, self).__init__()

        # Distortion Classifier (Model 1)
        self.model1 = models.mobilenet_v3_small(pretrained=False)
        self.model1.classifier[3] = nn.Linear(self.model1.classifier[3].in_features, 2)
        self.model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
        self.model1.eval()

        # Clean Face Recognizer (Model 2)
        self.model2 = models.mobilenet_v3_small(pretrained=False)
        num_classes_clean = self._infer_class_count(model2_path)
        self.model2.classifier[3] = nn.Linear(self.model2.classifier[3].in_features, num_classes_clean)
        self.model2.load_state_dict(torch.load(model2_path, map_location='cpu'))
        self.model2.eval()

        # Distorted Face Recognizer (Model 3)
        self.model3 = models.mobilenet_v3_small(pretrained=False)
        num_classes_distorted = self._infer_class_count(model3_path)
        self.model3.classifier[3] = nn.Linear(self.model3.classifier[3].in_features, num_classes_distorted)
        self.model3.load_state_dict(torch.load(model3_path, map_location='cpu'))
        self.model3.eval()

        # Class mapping for name lookup
        self.idx_to_class_clean = self._load_class_mapping(class_map_clean)
        self.idx_to_class_distorted = self._load_class_mapping(class_map_distorted)

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def _infer_class_count(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        return state_dict['classifier.3.bias'].shape[0]

    def _load_class_mapping(self, json_path):
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return idx_to_class
        return None

    def forward(self, img):
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            is_distorted = torch.argmax(self.model1(x)).item()

            if is_distorted == 0:  # Clean
                out = self.model2(x)
                idx = torch.argmax(out).item()
                label = self.idx_to_class_clean.get(idx, f'class_{idx}') if self.idx_to_class_clean else f'class_{idx}'
            else:  # Distorted
                out = self.model3(x)
                idx = torch.argmax(out).item()
                label = self.idx_to_class_distorted.get(idx, f'class_{idx}') if self.idx_to_class_distorted else f'class_{idx}'

            return label

    def evaluate_folder(self, folder_path, true_labels):
        preds, targets = [], []
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.png')):
               continue
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert("RGB")
            pred = self.forward(img)
            preds.append(pred)
            targets.append(true_labels.get(fname, 'unknown'))

        print("\n📊 Classification Report:")
        print(classification_report(targets, preds))

        print("\n📉 Confusion Matrix:")
        labels = sorted(set(targets + preds))  # All known labels
        cm = confusion_matrix(targets, preds, labels=labels)
        print(cm)

        # 🔥 Confusion Matrix Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()


# ================================
# ✅ Instantiate and Use
# ================================

model = UnifiedFaceAnalyzer(
    model1_path='/content/model1_distortion_classifier.pth',
    model2_path='/content/clean_mobilenetv3_face_recognition.pth',
    model3_path='/content/mobilenetv3_model_distorted_877.pth',
    class_map_clean='/content/class_to_idx_clean.json',  # replace if available
    class_map_distorted='/content/class_to_idx_877.json'
)

# ✅ Example: Inference on a single image
img = Image.open('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train/001_frontal/distortion/001_frontal_foggy.jpg').convert("RGB")
prediction = model(img)
print(f"🧠 Predicted Identity: {prediction}")

# ✅ Example: Batch Evaluation (optional)
# true_labels = {"person1.jpg": "John", "person2.jpg": "Alice"}  # true labels for evaluation
# model.evaluate_folder("/content/test", true_labels)

# ✅ Save model for GitHub
torch.save(model.state_dict(), "UnifiedFaceAnalyzer.pt")
print("✅ Unified model saved as UnifiedFaceAnalyzer.pt")