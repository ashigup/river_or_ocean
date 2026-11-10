from fastai.vision.all import *
import streamlit as st
from PIL import Image

# Load the exported model (cached so it only loads once)
@st.cache_resource
def load_model():
    return load_learner('export.pkl')

model = load_model()

st.title("Landscape Image Classifier 🌊🌲")
st.write("Upload an image to classify it using your fastai model!")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Classify Image'):
        with st.spinner('Analyzing...'):
            # Fastai prediction expects a PIL Image or filename
            pred, pred_idx, probs = model.predict(img)
            
            st.success(f"Prediction: **{pred}**")
            st.write(f"Confidence: **{probs[pred_idx]*100:.2f}%**")
