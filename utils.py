import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image, model):
    image = preprocess_image(image)
    preds = model.predict(image)
    confidence = float(preds[0][0] * 100)

    label = "Tumor" if confidence > 50 else "No Tumor"
    return label, confidence
