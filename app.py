import pathlib
import platform
import sys
import types
import traceback
from pathlib import Path
import streamlit as st
import torch
from fastai.vision.all import PILImage

# Compatibility shim: Kaggle uses Python 3.13 (pathlib._local), Streamlit uses Python 3.11
if "pathlib._local" not in sys.modules:
    pathlib_local = types.ModuleType("pathlib._local")
    pathlib_local.Path = pathlib.Path
    pathlib_local.PosixPath = pathlib.PosixPath
    pathlib_local.WindowsPath = pathlib.PosixPath if platform.system() != "Windows" else pathlib.WindowsPath
    pathlib_local.PurePath = pathlib.PurePath
    pathlib_local.PurePosixPath = pathlib.PurePosixPath
    pathlib_local.PureWindowsPath = pathlib.PureWindowsPath
    sys.modules["pathlib._local"] = pathlib_local

if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="River or Ocean Classifier", page_icon="🌊")

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        repo_files = [f.name for f in Path(__file__).parent.iterdir()]
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Available files: {repo_files}"
        )

    if MODEL_PATH.stat().st_size < 1024:
        raise ValueError(
            f"model.pkl is only {MODEL_PATH.stat().st_size} bytes (likely a Git LFS pointer instead of the actual model weights)."
        )

    learn = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
    learn.dls.cpu()
    learn.model.cpu()
    learn.model.eval()
    return learn


st.title("River or Ocean Image Classifier 🌊🏞️")

try:
    model = load_model()
except Exception as e:
    print("=== MODEL LOADING ERROR ===", flush=True)
    traceback.print_exc()
    st.header("⚠️ Error Loading Model")
    st.error(f"Could not initialize the classifier: {e}")
    st.stop()

st.write("Upload an image to classify whether it is a **river**, **ocean**, or **random**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            pred_class, pred_idx, probs = model.predict(img)

        st.success(f"**Prediction:** {pred_class.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]:.2%}")

        st.subheader("Class Probabilities")
        for label, prob in zip(model.dls.vocab, probs):
            st.write(f"- **{label}**: {prob:.2%}")

    except Exception as e:
        print("=== PREDICTION ERROR ===", flush=True)
        traceback.print_exc()
        st.header("⚠️ Error Processing Image")
        st.error(f"An error occurred while analyzing the image: {e}")
