import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

MODEL_PATH = "models/skin_cnn_model.h5"
@st.cache_resource
def load_skin_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error("Model file not found. Please add 'skin_cnn_model.h5' in models folder.")
        return None

model = load_skin_model()
CLASS_NAMES = ["Acne", "Eczema", "Psoriasis", "Ringworm", "Healthy"]

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🧴 Skin Disease Classifier (AI-Powered)")
st.markdown("Upload a close-up image of the affected skin area to detect potential conditions.")

uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            prediction = model.predict(img_array)[0]
            pred_idx = np.argmax(prediction)
            confidence = prediction[pred_idx]

            st.subheader(f"🩺 Predicted Condition: {CLASS_NAMES[pred_idx]}")
            st.write(f"Confidence: {confidence:.2%}")

            if CLASS_NAMES[pred_idx] == "Healthy":
                st.success("Your skin appears healthy. No issues detected.")
            else:
                st.warning(f"Please consult a dermatologist for further evaluation of {CLASS_NAMES[pred_idx]}.")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

st.markdown("---")
st.caption("⚠️ This is an AI demo. For serious skin conditions, consult a certified doctor.")
