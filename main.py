#all 3 models
# split training set into clean and distorted folder
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

print("✅ All clean and distorted images have been flattened into separate folders.")




#removen any broken file if exist
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

print(f"✅ Removed {len(broken_images)} broken files.")



#reproducebilty
import random
import numpy as np
import torch

# ✅ Set all relevant seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Slower but reproducible
    torch.backends.cudnn.benchmark = False

set_seed(42)


#model1_distortion_classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Paths
train_dir = "/content/drive/MyDrive/comys/Comys_Hackathon5/train_distortion"

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Dataset
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# ✅ Model: MobileNetV3 Small
model = models.mobilenet_v3_small(pretrained=True)
# Modify the entire classifier to accept 576 features
model.classifier = nn.Sequential(
    nn.Linear(576, 1024), # Add a linear layer to match the original classifier's expected input size
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(1024, 2), # Output 2 classes for distortion
)
model = model.to(device)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training Loop
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
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} Accuracy: {accuracy:.2f}%")

# ✅ Save Model
torch.save(model.state_dict(), "/content/model1_distortion_classifier.pth")
print("🎉 Model 1 saved successfully!")


import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("/content/model1_distortion_classifier.pth"))
model.to(device)
print("✅ Model loaded successfully!")



#Face Recognition (Clean Images)
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import json


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

print(f"🧑 Total Classes: {len(dataset.class_to_idx)}")
print("✅ Sample Classes:", list(dataset.class_to_idx.keys())[:5])



# Save class_to_idx mapping to JSON
with open('/content/class_to_idx_clean.json', 'w') as f:
    json.dump(dataset.class_to_idx, f)

print("✅ Saved: class_to_idx_clean.json")


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
epochs = 20
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
print("✅ Saved: clean_mobilenetv3_face_recognition.pth")


import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 877) # Corrected to 877 output features
model.load_state_dict(torch.load("/content/clean_mobilenetv3_face_recognition.pth"))
model.to(device)
print("✅ Model loaded successfully!")



#Face Recognition (Distorted Images)
import os
import random
import json
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# ===========================
# 🔐 Set Seed for Reproducibility
# ===========================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===========================
# 📂 Custom Dataset Class
# ===========================
class FlatDistortedFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, top_n_classes=None, split='train', split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        class_counter = 0

        # Group images by person name (from filename prefix)
        grouped = defaultdict(list)
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                person_name = filename.split("_distorted_")[0]
                grouped[person_name].append(filename)

        # Use all classes or top-N only
        if top_n_classes is not None:
            sorted_classes = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_classes]
        else:
            sorted_classes = grouped.items()

        for class_name, filenames in sorted_classes:
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = class_counter
                self.idx_to_class[class_counter] = class_name
                class_counter += 1

            all_paths = [os.path.join(root_dir, fname) for fname in filenames]
            random.shuffle(all_paths)

            split_idx = int(len(all_paths) * split_ratio)
            selected_paths = all_paths[:split_idx] if split == 'train' else all_paths[split_idx:]

            for path in selected_paths:
                self.samples.append((path, self.class_to_idx[class_name]))

        print(f"[{split.upper()}] Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ===========================
# 🧠 Training Distorted Face Recognition Model
# ===========================
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

# Dataset and DataLoaders (All 877 classes)
batch_size = 32

train_dataset = FlatDistortedFaceDataset(distorted_root, transform=train_transform, top_n_classes=None, split='train')
val_dataset = FlatDistortedFaceDataset(distorted_root, transform=val_transform, top_n_classes=None, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Save class mapping
with open('/content/class_to_idx_877.json', 'w') as f:
    json.dump(train_dataset.class_to_idx, f)

# ===========================
#  Model Setup
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.class_to_idx)

model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

# ===========================
# ⚙️ Training Setup
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===========================
# 🔁 Training Loop
# ===========================
epochs = 5
for epoch in range(epochs):
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
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f" Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}%")

    # 🔍 Validation
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

# ===========================
#  Save Trained Model
# ===========================
torch.save(model.state_dict(), '/content/mobilenetv3_model_distorted_877.pth')
print("✅ Model saved to: /content/mobilenetv3_model_distorted_877.pth")


import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 877) # Corrected to 100 output features
model.load_state_dict(torch.load("/content/mobilenetv3_model_distorted_877.pth"))
model.to(device)
print("✅ Model loaded successfully!")


