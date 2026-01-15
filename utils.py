import tensorflow as tf
import numpy as np
import cv2

# Load model architecture using tf.keras
with open("model.json", "r") as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights
model.load_weights("model.h5")

CLASSES = ["No Tumor", "Tumor"]

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    idx = np.argmax(prediction)
    return CLASSES[idx], prediction[idx] * 100
