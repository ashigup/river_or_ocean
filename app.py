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

st.set_page_config(
    page_title="River or Ocean Classifier",
    page_icon="🌊",
    layout="centered",
)

# Custom CSS for cleaner cards and footer styling
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        font-size: 0.9rem;
        color: #6c757d;
    }
    .footer a {
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).parent / "model.pkl"
GITHUB_URL = "https://github.com/ashigup/river_or_ocean"

CLASS_META = {
    "river": ("🏞️", "River"),
    "ocean": ("🌊", "Ocean"),
    "random": ("🎲", "Random / Other"),
}

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


# Sidebar info
with st.sidebar:
    st.header("About This App")
    st.write(
        "A computer vision classifier trained with **fastai** & **PyTorch** (`resnet18`) "
        "to distinguish between **rivers**, **oceans**, and **random** scenes."
    )
    st.divider()
    st.markdown(f"**Built by:** Ashish Kumar")
    st.markdown(f"🔗 [View Source on GitHub]({GITHUB_URL})")

# Header
st.markdown("<h1 class='main-header'>🌊 River or Ocean Classifier 🏞️</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Upload any landscape photo to see whether the model recognizes a <b>River</b>, an <b>Ocean</b>, or <b>Random</b> scenery.</p>",
    unsafe_allow_html=True,
)

try:
    net, vocab = load_model()
except Exception as e:
    print("=== MODEL LOADING ERROR ===", flush=True)
    traceback.print_exc()
    st.header("⚠️ Error Loading Model")
    st.error(f"Could not initialize the classifier: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP",
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing image..."):
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                logits = net(input_tensor)
                probs = F.softmax(logits, dim=1)[0]

            pred_idx = int(torch.argmax(probs).item())
            pred_raw = vocab[pred_idx]
            emoji, pred_label = CLASS_META.get(pred_raw.lower(), ("🔍", pred_raw.capitalize()))
            confidence = probs[pred_idx].item()

        st.divider()
        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.image(img, caption=uploaded_file.name, use_container_width=True)

        with col2:
            st.subheader("Result")
            st.metric(label="Top Prediction", value=f"{emoji} {pred_label}", delta=f"{confidence:.1%} confidence")

            st.markdown("#### Class Breakdown")
            # Sort classes from highest to lowest probability
            ranked = sorted(
                zip(vocab, [p.item() for p in probs]),
                key=lambda x: x[1],
                reverse=True,
            )
            for label, prob in ranked:
                item_emoji, item_name = CLASS_META.get(label.lower(), ("•", label.capitalize()))
                st.write(f"**{item_emoji} {item_name}** — `{prob:.2%}`")
                st.progress(min(max(prob, 0.0), 1.0))

    except Exception as e:
        print("=== PREDICTION ERROR ===", flush=True)
        traceback.print_exc()
        st.header("⚠️ Error Processing Image")
        st.error(f"An error occurred while analyzing the image: {e}")
else:
    st.info("👆 Upload an image above to get started.")

# Footer with author credit and GitHub link
st.markdown(
    f"""
    <div class="footer">
        Built by <b>Ashish Kumar</b>  • 
        <a href="{GITHUB_URL}" target="_blank">GitHub Repository ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
