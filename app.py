import io

import streamlit as st
import torch
from PIL import Image

from feature_extraction import load_feature_extractor
from preprocessing import denormalize_image, preprocess_image
from style_transfer import transfer_style

st.set_page_config(page_title="ColourMorph", layout="wide")
st.title("ColourMorph")
st.caption(
    "The app keeps the scene of the content image and applies the colours and texture of the style image."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_model(device):
    return load_feature_extractor(device=device)


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

steps = st.slider("Steps", min_value=1, max_value=500, value=300)
generate = st.button("Generate", disabled=content_file is None or style_file is None)

if "result_png" not in st.session_state:
    st.session_state.result_png = None

if generate:
    progress = st.progress(0, text="Starting")

    def on_step(step, total_steps, loss):
        progress.progress(step / total_steps, text=f"Step {step}/{total_steps}")

    try:
        content = preprocess_image(open_upload(content_file), device=DEVICE)
        style = preprocess_image(open_upload(style_file), device=DEVICE)
        model = get_model(DEVICE)
        output = transfer_style(
            content,
            style,
            model,
            {"steps": steps, "callback": on_step},
        )
        result = denormalize_image(output)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        st.session_state.result_png = buffer.getvalue()
        progress.progress(1.0, text=f"Step {steps}/{steps}")
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
