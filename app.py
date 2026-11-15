import pickle, traceback, os
import torch
from fastai.vision.all import *
import streamlit as st
from PIL import Image

st.write("export.pkl exists:", os.path.exists("export.pkl"))
if os.path.exists("export.pkl"):
    st.write("size (bytes):", os.path.getsize("export.pkl"))
    with open("export.pkl", "rb") as f:
        st.write("first bytes:", f.read(60))

try:
    raw = torch.load("export.pkl", map_location="cpu", pickle_module=pickle, weights_only=False)
    st.success("torch.load worked")
    st.write(type(raw))
except BaseException:
    st.code(traceback.format_exc())
    st.stop()

@st.cache_resource
def load_model():
    return load_learner("export.pkl", cpu=True)

model = load_model()
