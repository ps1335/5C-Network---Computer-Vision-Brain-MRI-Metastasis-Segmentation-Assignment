import tensorflow as tf
from tensorflow.keras import layers, models

# Nested U-Net (U-Net++)
def nested_unet(input_shape):
    inputs = layers.Input(input_shape)
    # Define your U-Net++ architecture
    # For brevity, the code is simplified
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(inputs)
    model = models.Model(inputs, outputs)
    return model

# Attention U-Net
def attention_unet(input_shape):
    inputs = layers.Input(input_shape)
    # Define your Attention U-Net architecture
    # For brevity, the code is simplified
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(inputs)
    model = models.Model(inputs, outputs)
    return model
