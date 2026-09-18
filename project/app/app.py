import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/plant_disease_model.keras')

@st.cache_resource
def load_disease_info():
    with open('data/plant_disease_info.json') as f:
        return json.load(f)

model = load_model()
disease_info = load_disease_info()
class_names = sorted(disease_info.keys())

st.title("🌿 Plant Disease Detector")
st.write("Upload a photo of a plant leaf to check for possible diseases.")
st.caption("Trained on: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, "
           "Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")

uploaded_file = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded photo", use_container_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    info = disease_info[predicted_class]

    st.subheader(f"Crop: {info['crop']}")
    st.subheader(f"Diagnosis: {info['disease']} ({info['type']})")
    st.write(f"Confidence: {confidence:.1f}%")

    if confidence < 60:
        st.warning(
            "Confidence is fairly low — this photo may not match one of the "
            "14 crop species this model was trained on. Results may be unreliable."
        )

    st.write(info['description'])

    if info['prevention']:
        st.markdown("**Prevention:**")
        for tip in info['prevention']:
            st.write(f"- {tip}")

    if info['treatment']:
        st.markdown("**Treatment:**")
        for tip in info['treatment']:
            st.write(f"- {tip}")
    else:
        st.success("No treatment needed — plant appears healthy!")
