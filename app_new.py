import hashlib
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageDraw

import image_tools as it
from feature_extraction import load_feature_extractor
from preprocessing import denormalize_image, preprocess_image
from style_transfer import PILOT_IMAGE_SIZE, transfer_style, load_decoder
from ui import parts

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_decoder.pth"
SAMPLE_DIR = ROOT / "sample_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_LABEL = "GPU" if DEVICE == "cuda" else "CPU"

# The decoder was trained at PILOT_IMAGE_SIZE. "Sharper" runs the same decoder on
# a larger canvas, which keeps more fine detail but is less tested
DETAIL_LEVELS = {"Standard": PILOT_IMAGE_SIZE, "Sharper": PILOT_IMAGE_SIZE * 2}


def make_page_icon():
    icon = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    for (x, y), colour in zip([(4, 4), (34, 4), (4, 34), (34, 34)], ["#3B4BDB", "#E4577A", "#19A58A", "#F4B63F"]):
        draw.ellipse((x, y, x + 26, y + 26), fill=colour)
    return icon


st.set_page_config(page_title="ColourMorph",
                   page_icon=make_page_icon(), layout="wide")


@st.cache_resource
def get_models(device, checkpoint_path):
    extractor = load_feature_extractor(device=device)
    decoder = load_decoder(checkpoint_path, device=device)
    return extractor, decoder


@st.cache_data(show_spinner=False, max_entries=24)
def prepare_preview(data: bytes):
    """Decode once per upload. Slider moves rerun the script, so this is cached."""
    try:
        image = it.open_image(data)
    except it.ImageReadError as exc:
        return {"error": str(exc)}
    preview = it.fit_long_side(image, it.PREVIEW_LONG_SIDE)
    return {
        "uri": it.to_data_uri(preview, quality=88),
        "size": it.describe_size(image),
        "palette": it.extract_palette(image),
    }


def first_sample(kind):
    folder = SAMPLE_DIR / kind
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
    return None


def pick_source(uploaded, kind):
    """Return (bytes, name, is_sample). An upload always wins over a sample."""
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name, False
    if st.session_state.get("use_samples"):
        path = first_sample(kind)
        if path is not None:
            return path.read_bytes(), path.name, True
    return None, None, False


def strength_note(value):
    if value == 0:
        return "Your original canvas, untouched"
    if value <= 30:
        return "A gentle hint of artistry while your canvas keeps its original character"
    if value <= 70:
        return "A meeting of moods as the artwork and your canvas find their balance"
    return "Find the perfect balance between your original photograph and artistic expression"


def set_samples(value):
    st.session_state.use_samples = value
    clear_result()


def clear_result():
    st.session_state.result = None


def render_slot(title, hint, kind, empty_message, palette_label, key):
    st.markdown(parts.slot_heading(title, hint), unsafe_allow_html=True)
    uploaded = st.file_uploader(
        title, type=["jpg", "jpeg", "png"], key=key, label_visibility="collapsed",
        max_upload_size=50, on_change=clear_result,
    )
    data, name, is_sample = pick_source(uploaded, kind)
    if data is None:
        st.markdown(parts.empty_frame(empty_message), unsafe_allow_html=True)
        st.markdown(parts.swatches([], ""), unsafe_allow_html=True)
        return None
    info = prepare_preview(data)
    if "error" in info:
        st.markdown(parts.error_frame(info["error"]), unsafe_allow_html=True)
        st.markdown(parts.swatches([], ""), unsafe_allow_html=True)
        return None
    st.markdown(
        parts.frame(info["uri"], name, info["size"],
                    f"{title} preview", "Sample" if is_sample else ""),
        unsafe_allow_html=True,
    )
    st.markdown(parts.swatches(info["palette"],
                palette_label), unsafe_allow_html=True)
    return {
        "data": data,
        "name": name,
        "palette": info["palette"],
        "is_sample": is_sample,
    }


# ---------------------------------------------------------------- page top
model_ready = CHECKPOINT.is_file()
header_col, theme_col = st.columns([5, 1], vertical_alignment="center")
with header_col:
    st.markdown(parts.header(model_ready, DEVICE_LABEL), unsafe_allow_html=True)
with theme_col:
    dark_mode = st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between light and dark appearance.",
    )
st.markdown(
    f"<style>{parts.load_css(dark_mode=dark_mode)}</style>",
    unsafe_allow_html=True,
)

content_col, style_col = st.columns(2, gap="large")
with content_col:
    content = render_slot(
        "Your Canvas", "Your story stays untouched while its visual world transforms", "content",
        "Drop in the photo you want to transform", "Colours in your photo", "content_upload",
    )
with style_col:
    style = render_slot(
        "Your Artistic Muse",
        "Discover the palette and textures that will inspire your transformation",
        "style",
        "Drop in a painting whose look you love", "Colours it will lend", "style_upload",
    )

has_samples = first_sample("content") is not None and first_sample("style") is not None
sample_active = bool(
    (content and content["is_sample"]) or (style and style["is_sample"])
)
if has_samples:
    if sample_active:
        st.button(
            "Clear the Canvas", on_click=set_samples, args=(False,), key="samples_off"
        )
    else:
        st.button(
            "Try the sample images", on_click=set_samples, args=(True,), key="samples_on"
        )

# ---------------------------------------------------------------- controls
with st.container(key="controls"):
    strength_col, detail_col, action_col = st.columns(
        [1.3, 1, 0.8], gap="large", vertical_alignment="bottom")
    with strength_col:
        strength = st.slider(
            "Artistic Influence", min_value=0, max_value=100, value=100, step=5, format="%d%%",
            help="0% leaves your photo untouched. 100% repaints it fully in the style image's look.",
        )
        st.markdown(
            f'<p class="cm-help">{strength_note(strength)}</p>', unsafe_allow_html=True)
    with detail_col:
        detail = st.segmented_control(
            "Artistic Precision", list(DETAIL_LEVELS), default="Standard", key="detail",
            help="Standard is fastest and matches how the model was trained. Sharper keeps finer texture and takes a little longer.",
        ) or "Standard"
    ready = content is not None and style is not None and model_ready
    with action_col:
        generate = st.button("Bring Art to Life", type="primary",
                             disabled=not ready, width="stretch")
    if not model_ready:
        pass
    elif content is None and style is None:
        st.markdown(
            '<p class="cm-help">Add a content image and a style image to begin.</p>', unsafe_allow_html=True)
    elif content is None:
        st.markdown(
            '<p class="cm-help">Add a content image to continue.</p>', unsafe_allow_html=True)
    elif style is None:
        st.markdown(
            '<p class="cm-help">Add a style image to continue.</p>', unsafe_allow_html=True)

if not model_ready:
    st.error(
        "No trained decoder at checkpoints/best_decoder.pth. "
        "Run git pull to get the shared checkpoint, or train one with python train.py "
        "after placing images in data/content/train and data/style/train."
    )

# ---------------------------------------------------------------- generate
if "result" not in st.session_state:
    st.session_state.result = None


def settings_key():
    return (
        hashlib.md5(content["data"]).hexdigest(),
        hashlib.md5(style["data"]).hexdigest(),
        strength,
        detail,
    )


if generate:
    try:
        with st.spinner("Applying the style. This takes a few seconds."):
            started = time.perf_counter()
            content_image = it.open_image(content["data"])
            style_image = it.open_image(style["data"])
            canvas = DETAIL_LEVELS[detail]
            content_tensor = preprocess_image(
                content_image, image_size=canvas, device=DEVICE)
            style_tensor = preprocess_image(
                style_image, image_size=canvas, device=DEVICE)
            extractor, decoder = get_models(DEVICE, str(CHECKPOINT))
            output = transfer_style(
                content_tensor,
                style_tensor,
                alpha=strength / 100,
                feature_extractor=extractor,
                decoder=decoder,
                device=DEVICE,
            )
            raw = denormalize_image(output)
            result_image, original_image = it.finish_result(
                raw, content_image, mix=strength / 100)
            elapsed = time.perf_counter() - started
        st.session_state.result = {
            "key": settings_key(),
            "before_uri": it.to_data_uri(original_image, quality=92),
            "after_uri": it.to_data_uri(result_image, quality=92),
            "png": it.encode(result_image, "PNG"),
            "jpg": it.encode(result_image, "JPEG"),
            "size": it.describe_size(result_image),
            "strength": strength,
            "detail": detail,
            "seconds": elapsed,
            "palette": it.extract_palette(result_image),
            "style_palette": style["palette"],
        }
    except Exception as exc:
        st.session_state.result = None
        st.error(f"Could not generate the image: {exc}")

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------- result
result = st.session_state.result
if result is not None:
    st.markdown(
        '<div class="cm-result-head"><h2>A Tale of Two Perspectives</h2>'
        "<p>Move the slider and uncover the artistic side of your photograph</p></div>",
        unsafe_allow_html=True,
    )
    if ready and result["key"] != settings_key():
        st.markdown(
            '<div class="cm-stale">The images or settings have changed since this result. '
            "Select Bring Art to Life to refresh your creation</div>",
            unsafe_allow_html=True,
        )
    view_col, side_col = st.columns([2.1, 1], gap="large")
    with view_col:
        components.html(parts.compare_slider(
            result["before_uri"], result["after_uri"], dark_mode=dark_mode), height=480)
    with side_col:
        with st.container(key="resultside"):
            st.markdown(
                parts.result_facts(
                    result["size"], result["strength"], result["detail"], result["seconds"], DEVICE_LABEL)
                + parts.swatches(result["style_palette"], "Style colours")
                + parts.swatches(result["palette"], "Result colours"),
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download PNG", data=result["png"], file_name="colourmorph_result.png",
                mime="image/png", type="primary", width="stretch",
            )
            st.download_button(
                "Download JPG", data=result["jpg"], file_name="colourmorph_result.jpg",
                mime="image/jpeg", width="stretch",
            )

# ---------------------------------------------------------------- creative process
animate_steps = not st.session_state.get("how_steps_seen", False)
st.session_state.how_steps_seen = True
st.markdown(parts.how_it_works(animate=animate_steps), unsafe_allow_html=True)
