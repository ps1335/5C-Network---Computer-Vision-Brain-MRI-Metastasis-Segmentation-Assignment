import numpy as np
import tensorflow as tf
from sklearn.metrics import jaccard_score
from data_preprocessing import preprocess_and_split_data
from models import nested_unet, attention_unet

# Dice coefficient calculation
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Train the models and evaluate
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=8):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Load data
image_dir = r'D:\Data\images'
mask_dir = r'D:\Data\masks'
X_train, X_test, y_train, y_test = preprocess_and_split_data(image_dir, mask_dir)

# Train both models and compare
input_shape = X_train.shape[1:]

# Nested U-Net
nested_model = nested_unet(input_shape)
train_and_evaluate(nested_model, X_train, y_train, X_test, y_test)

# Attention U-Net
attention_model = attention_unet(input_shape)
train_and_evaluate(attention_model, X_train, y_train, X_test, y_test)
