import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Apply CLAHE to an image
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Load images and masks, apply preprocessing (CLAHE)
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        if not os.path.exists(mask_path):
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        img = apply_clahe(img)
        img = img / 255.0  # Normalize image
        
        mask = mask / 255.0  # Normalize mask (binary mask)
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def preprocess_and_split_data(image_dir, mask_dir, test_size=0.2):
    images, masks = load_data(image_dir, mask_dir)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Define your data directories
image_dir = r'D:\Data\images'
mask_dir = r'D:\Data\masks'

# Preprocess and split data
X_train, X_test, y_train, y_test = preprocess_and_split_data(image_dir, mask_dir)
