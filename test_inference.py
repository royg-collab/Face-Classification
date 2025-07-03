
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('multi_model_face_recognizer.pt', map_location='cpu')
model.eval()

# Define validation directory
val_dir = 'val'
class_names = sorted(os.listdir(val_dir))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

y_true = []
y_pred = []

for label_name in class_names:
    label = class_to_idx[label_name]
    img_folder = os.path.join(val_dir, label_name)

    for file in os.listdir(img_folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(img_folder, file)
            pred = model.predict_image(img_path)
            y_true.append(label)
            y_pred.append(pred)

# Generate report
print(classification_report(y_true, y_pred, target_names=class_names))
