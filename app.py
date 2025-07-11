import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from googletrans import Translator

translator = Translator()

def translate_text(text, lang):
    try:
        translated = translator.translate(text, dest=lang)
        return translated.text
    except Exception as e:
        return text


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
lang_option = st.selectbox("Choose Language", ["English", "Hindi", "Marathi"])
lang_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
selected_lang = lang_map[lang_option]

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

            st.subheader(translate_text(f"🩺 Predicted Condition: {CLASS_NAMES[pred_idx]}", selected_lang))
st.write(translate_text(f"Confidence: {confidence:.2%}", selected_lang))

if CLASS_NAMES[pred_idx] == "Healthy":
    st.success(translate_text("Your skin appears healthy. No issues detected.", selected_lang))
else:
    st.warning(translate_text(f"Please consult a dermatologist for further evaluation of {CLASS_NAMES[pred_idx]}.", selected_lang))


    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

st.markdown("---")
st.caption("⚠️ This is an AI demo. For serious skin conditions, consult a certified doctor.")
