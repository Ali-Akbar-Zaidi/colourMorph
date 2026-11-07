import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from feature_extraction import load_feature_extractor
from preprocessing import denormalize_image, preprocess_image
from style_transfer import PILOT_IMAGE_SIZE, transfer_style, load_decoder

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_decoder.pth"

st.set_page_config(page_title="ColourMorph", layout="wide")
st.title("ColourMorph")
st.caption(
    "The app keeps the scene of the content image and applies the colours and texture of the style image."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_models(device, checkpoint_path):
    extractor = load_feature_extractor(device=device)
    decoder = load_decoder(checkpoint_path, device=device)
    return extractor, decoder


def open_upload(uploaded):
    return Image.open(io.BytesIO(uploaded.getvalue()))


content_col, style_col = st.columns(2)
with content_col:
    content_file = st.file_uploader("Content image", type=["jpg", "jpeg", "png"])
    if content_file is not None:
        st.image(open_upload(content_file), caption="Content preview", width="stretch")
with style_col:
    style_file = st.file_uploader("Style image", type=["jpg", "jpeg", "png"])
    if style_file is not None:
        st.image(open_upload(style_file), caption="Style preview", width="stretch")

alpha = st.slider("Style strength", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
generate = st.button(
    "Generate",
    disabled=content_file is None or style_file is None or not CHECKPOINT.is_file(),
)

if not CHECKPOINT.is_file():
    st.error(
        "No trained decoder at checkpoints/best_decoder.pth. "
        "Train one with python train.py after placing images in data/content/train and data/style/train."
    )

if "result_png" not in st.session_state:
    st.session_state.result_png = None

if generate:
    try:
        content = preprocess_image(open_upload(content_file), image_size=PILOT_IMAGE_SIZE, device=DEVICE)
        style = preprocess_image(open_upload(style_file), image_size=PILOT_IMAGE_SIZE, device=DEVICE)
        extractor, decoder = get_models(DEVICE, str(CHECKPOINT))
        output = transfer_style(
            content,
            style,
            alpha=alpha,
            feature_extractor=extractor,
            decoder=decoder,
            device=DEVICE,
        )
        result = denormalize_image(output)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        st.session_state.result_png = buffer.getvalue()
    except Exception as exc:
        st.session_state.result_png = None
        st.error(f"Could not generate the image: {exc}")

if st.session_state.result_png is not None:
    st.image(st.session_state.result_png, caption="Result", width="stretch")
    st.download_button(
        "Download",
        data=st.session_state.result_png,
        file_name="colourmorph_result.png",
        mime="image/png",
    )
