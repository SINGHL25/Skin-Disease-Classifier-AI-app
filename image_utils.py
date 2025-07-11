# skin_checker_app/utils/image_utils.py

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

def preprocess_image(uploaded_file, target_size=(224, 224)):
    """
    Load and preprocess an uploaded image.
    Returns: processed image array or None if error occurs.
    """
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize(target_size)
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image, image_array
    except Exception as e:
        return None, f"Error processing image: {e}"

def predict_image(model, image_array, class_names):
    """
    Predicts the class of the image using the loaded model.
    Returns: (predicted_class_name, confidence_score)
    """
    try:
        prediction = model.predict(image_array)[0]
        pred_idx = np.argmax(prediction)
        confidence = prediction[pred_idx]
        return class_names[pred_idx], confidence
    except Exception as e:
        return "Unknown", f"Prediction error: {e}"

