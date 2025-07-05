#test
import shutil
import os

# Paths to test images
test_images = {
    "001_frontal_foggy.jpg": "001_frontal",
    "002_frontal_blur.jpg": "002_frontal"
}

# Create a flat test directory
os.makedirs("/content/test_images", exist_ok=True)

# Copy the images
shutil.copy('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train/001_frontal/001_frontal.jpg',
            '/content/test_images/001_frontal_clear.jpg')

# Fix: Copy a specific file from the distortion directory, not the directory itself
shutil.copy('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train/001_frontal/distortion/001_frontal_foggy.jpg',
            '/content/test_images/001_frontal_foggy.jpg')

# Add more test images
shutil.copy('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train/002_frontal/distortion/002_frontal_blurred.jpg',
            '/content/test_images/002_frontal_blur.jpg')
shutil.copy('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_B/train/001_frontal/distortion/001_frontal_rainy.jpg',
            '/content/test_images/001_frontal_rainy.jpg')


true_labels = {
    "001_frontal_foggy.jpg": "001_frontal",
    "001_frontal_clear.jpg": "001_frontal",
    "002_frontal_blur.jpg": "002_frontal",
    "001_frontal_rainy.jpg": "001_frontal", # Corrected key name
    "001_frontal_cloudy.jpg": "001_frontal" # Assuming this image exists for testing
}

model.evaluate_folder("/content/test_images", true_labels)