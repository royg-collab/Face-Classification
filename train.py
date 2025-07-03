# split train data into clean and distorted folder
import os
import shutil

# Original train path
source_train = '/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train'

# Target clean/distorted folders
target_clean = '/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion/clean'
target_distorted = '/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion/distorted'

# Create target folders
os.makedirs(target_clean, exist_ok=True)
os.makedirs(target_distorted, exist_ok=True)

# Go through each person folder
for folder_name in os.listdir(source_train):
    folder_path = os.path.join(source_train, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Copy the clean/original image
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(target_clean, f"{folder_name}_{file}"))

    # Handle distortion folder
    distortion_folder = os.path.join(folder_path, 'distortion')
    if os.path.exists(distortion_folder):
        for file in os.listdir(distortion_folder):
            file_path = os.path.join(distortion_folder, file)
            if file.endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(file_path):
                new_name = f"{folder_name}_distorted_{file}"
                shutil.copy(file_path, os.path.join(target_distorted, new_name))

print(" All clean and distorted images have been flattened into separate folders.")


# remove broken image
from PIL import Image
import os

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False

# Check in distorted folder
distorted_path = '/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion/distorted'
broken_images = []

for fname in os.listdir(distorted_path):
    fpath = os.path.join(distorted_path, fname)
    if not is_valid_image(fpath):
        print(f"❌ Corrupted: {fname}")
        broken_images.append(fpath)

# Optional: delete broken images
for f in broken_images:
    os.remove(f)

print(f" Removed {len(broken_images)} broken files.")


#model_distortion_classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

#  Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Paths
train_dir = "/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion"

#  Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#  Dataset
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

#  Model: MobileNetV3 Small
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 2 classes: clean/distorted
model = model.to(device)

#  Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#  Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f" Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} Accuracy: {accuracy:.2f}%")

#  Save Model
torch.save(model, "/content/model1_distortion_classifier.pth")
print("🎉 Model 1 saved successfully!")

# model2_clean_face recognigition
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re

# Custom Dataset Class
class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

        # Extract class name from filename
        self.class_names = sorted(list(set([self._get_class_id(p) for p in self.image_paths])))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

    def _get_class_id(self, path):
        fname = os.path.basename(path)
        match = re.match(r'([^_]+_[^_]+)', fname)
        return match.group(1) if match else "unknown"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_name = self._get_class_id(img_path)
        label = self.class_to_idx[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths
clean_dir = '/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion/clean'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and Loader
dataset = FaceRecognitionDataset(clean_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

print(f" Total Classes: {len(dataset.class_to_idx)}")
print(" Sample Classes:", list(dataset.class_to_idx.keys())[:5])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV3-Small with SE
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(dataset.class_to_idx))  # Update final FC layer
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f" Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")

# Save Model
torch.save(model.state_dict(), '/content/clean_mobilenetv3_face_recognition.pth')
print(" Saved: clean_mobilenetv3_face_recognition.pth")


# model3_distorted_face recognition
import os
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import random

class FlatDistortedFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, top_n_classes=100, split='train', split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        class_counter = 0

        # Group images by class name
        grouped = defaultdict(list)
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                person_name = filename.split("_distorted_")[0]
                grouped[person_name].append(filename)

        # Keep only top N classes with most images
        sorted_classes = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_classes]

        for class_name, filenames in sorted_classes:
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = class_counter
                self.idx_to_class[class_counter] = class_name
                class_counter += 1

            all_paths = [os.path.join(root_dir, fname) for fname in filenames]
            random.shuffle(all_paths)

            split_idx = int(len(all_paths) * split_ratio)
            if split == 'train':
                selected_paths = all_paths[:split_idx]
            else:
                selected_paths = all_paths[split_idx:]

            for path in selected_paths:
                self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

# Paths
distorted_root = '/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion/distorted'

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset & DataLoaders
top_n_classes = 100
batch_size = 32

train_dataset = FlatDistortedFaceDataset(distorted_root, transform=train_transform, top_n_classes=top_n_classes, split='train')
val_dataset = FlatDistortedFaceDataset(distorted_root, transform=val_transform, top_n_classes=top_n_classes, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, top_n_classes)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f" Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f" Validation Accuracy: {val_acc:.2f}%\n")

# Save model
torch.save(model.state_dict(), '/content/mobilenetv3_model3_distorted.pth')
print(" Saved model to: /content/mobilenetv3_model3_distorted.pth")


# multi_model_face_recognizer_model
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== CLASS DEFINITION ========== #
class MultiModelFaceRecognizer(nn.Module):
    def __init__(self, model1_path, model2_path, model3_path, num_classes):
        super().__init__()
        
        # Load distortion classifier (Model 1)
        self.model1 = torch.load(model1_path, map_location=device)
        self.model1.eval()

        # Model 2: Clean images
        self.model2 = models.mobilenet_v3_small(pretrained=False)
        self.model2.classifier[3] = nn.Linear(self.model2.classifier[3].in_features, num_classes)
        self.model2.load_state_dict(torch.load(model2_path, map_location=device))
        self.model2.eval()

        # Model 3: Distorted images
        self.model3 = models.mobilenet_v3_small(pretrained=False)
        self.model3.classifier[3] = nn.Linear(self.model3.classifier[3].in_features, num_classes)
        self.model3.load_state_dict(torch.load(model3_path, map_location=device))
        self.model3.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Distortion classification
            distortion_type = torch.argmax(self.model1(x).cpu(), dim=1).item()
            if distortion_type == 0:  # clean
                out = self.model2(x.to(device))
            else:
                out = self.model3(x.to(device))
            
            pred = torch.argmax(out, dim=1).item()
        return pred


# ========== Run and Save ========== #
val_dir = "/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/val"
model1_path = "/content/model1_distortion_classifier.pth"
model2_path = "/content/mobilenetv3_model2_clean.pth"
model3_path = "/content/mobilenetv3_model3_distorted.pth"
num_classes = 100  # Update based on your dataset

# Create class index mapping
person_folders = sorted([f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))])
label_map = {name: idx for idx, name in enumerate(person_folders)}
idx_to_name = {v: k for k, v in label_map.items()}

# Initialize unified model
multi_model = MultiModelFaceRecognizer(model1_path, model2_path, model3_path, num_classes)
multi_model.to(device)

# Inference
y_true, y_pred = [], []

for person in person_folders:
    label = label_map[person]
    person_folder = os.path.join(val_dir, person)

    # Original images
    for img in os.listdir(person_folder):
        if img.endswith('.jpg') and not img.startswith('._'):
            path = os.path.join(person_folder, img)
            pred = multi_model.predict_image(path)
            y_true.append(label)
            y_pred.append(pred)

    # Distorted images
    distorted_path = os.path.join(person_folder, "distortion")
    if os.path.exists(distorted_path):
        for img in os.listdir(distorted_path):
            if img.endswith('.jpg') and not img.startswith('._'):
                path = os.path.join(distorted_path, img)
                pred = multi_model.predict_image(path)
                y_true.append(label)
                y_pred.append(pred)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f" Total Accuracy: {acc*100:.2f}%")

# Report
print("\n🧾 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=person_folders))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=person_folders, yticklabels=person_folders, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the full pipeline
torch.save(multi_model, "/content/multi_model_face_recognizer.pt")
print(" Full model pipeline saved to /content/multi_model_face_recognizer.pt")


