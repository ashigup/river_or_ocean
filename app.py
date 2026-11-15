from fastai.vision.all import *
import streamlit as st
from PIL import Image

@st.cache_resource
def load_model():
    return load_learner('export.pkl', cpu=True)

model = load_model()

st.title("Landscape Image Classifier 🌊🌲")
st.write("Upload an image to classify it using your fastai model!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    if st.button('Classify Image'):
        with st.spinner('Analyzing...'):
            pred, pred_idx, probs = model.predict(PILImage.create(uploaded_file))
            st.success(f"Prediction: **{pred}**")
            st.write(f"Confidence: **{probs[pred_idx]*100:.2f}%**")
