import pathlib
import platform
import sys
import types
import traceback
from pathlib import Path
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

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

# Match the exact preprocessing from Kaggle: Resize(192, method='squish') + ImageNet stats
preprocess = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
    vocab = list(learn.dls.vocab)
    net = learn.model.cpu()
    net.eval()
    return net, vocab


st.title("River or Ocean Image Classifier 🌊🏞️")

try:
    net, vocab = load_model()
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
        # Convert to RGB so PNGs with transparency (4 channels) don't fail
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                logits = net(input_tensor)
                probs = F.softmax(logits, dim=1)[0]

            pred_idx = int(torch.argmax(probs).item())
            pred_class = vocab[pred_idx]

        st.success(f"**Prediction:** {pred_class.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]:.2%}")

        st.subheader("Class Probabilities")
        for label, prob in zip(vocab, probs):
            st.write(f"- **{label}**: {prob.item():.2%}")

    except Exception as e:
        print("=== PREDICTION ERROR ===", flush=True)
        traceback.print_exc()
        st.header("⚠️ Error Processing Image")
        st.error(f"An error occurred while analyzing the image: {e}")
