import pathlib
import platform
import streamlit as st
from fastai.vision.all import load_learner, PILImage

# Ensure Path compatibility across OS environments
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="River or Ocean Classifier", page_icon="🌊")

import torch

@st.cache_resource
def load_model():
    learn = torch.load('model.pkl', map_location=torch.device('cpu'), weights_only=False)
    learn.dls.cpu()
    learn.model.cpu()
    learn.model.eval()
    return learn

model = load_model()

st.title("River or Ocean Image Classifier 🌊🏞️")
st.write("Upload an image to classify whether it is a **river**, **ocean**, or **random**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    img = PILImage.create(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Make prediction
    with st.spinner("Analyzing image..."):
        pred_class, pred_idx, probs = model.predict(img)

    st.success(f"**Prediction:** {pred_class.capitalize()}")
    st.write(f"**Confidence:** {probs[pred_idx]:.2%}")

    # Show probability breakdown for all classes ('ocean', 'random', 'river')
    st.subheader("Class Probabilities")
    for label, prob in zip(model.dls.vocab, probs):
        st.write(f"- **{label}**: {prob:.2%}")
