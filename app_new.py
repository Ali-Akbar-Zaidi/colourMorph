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
STYLE_PRESETS = {"Custom": None, "Gentle": 30, "Balanced": 65, "Bold": 100}
VARIATION_STRENGTHS = (30, 65, 100)
MAX_HISTORY = 5

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


@st.dialog("Full-screen comparison", width="large")
def show_fullscreen_comparison(
    before_uri, after_uri, mode, zoom, dark_mode, difference_uri=None, style_uri=None
):
    components.html(
        parts.compare_slider(
            before_uri, after_uri, height=760, dark_mode=dark_mode,
            mode=mode, zoom=zoom, difference_uri=difference_uri,
            style_uri=style_uri,
        ),
        height=760,
        scrolling=False,
    )


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


@st.cache_data(show_spinner=False, max_entries=8)
def build_showcase_png(before_uri, result_png, palette, strength):
    showcase = it.create_showcase_image(
        it.image_from_data_uri(before_uri),
        it.open_image(result_png),
        palette,
        strength,
    )
    return it.encode(showcase, "PNG")


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


def apply_style_preset():
    value = STYLE_PRESETS[st.session_state.style_preset]
    if value is not None:
        st.session_state.artistic_influence = value
    clear_result()


def mark_custom_influence():
    st.session_state.style_preset = "Custom"
    clear_result()


def show_history_result(index):
    st.session_state.result = {
        **st.session_state.history[index],
        "history_view": True,
    }


def select_variation(strength):
    result = st.session_state.result
    variant = next(
        (item for item in result["variants"] if item["strength"] == strength),
        None,
    )
    if variant is None:
        raise ValueError(f"No {strength}% variation is available.")

    image = it.open_image(variant["png"])
    original = it.image_from_data_uri(result["before_uri"])
    selected = result.copy()
    selected.update(
        {
            "strength": strength,
            "key": (*result["key"][:2], strength, result["key"][3]),
            "after_uri": it.to_data_uri(image, quality=92),
            "png": variant["png"],
            "jpg": it.encode(image, "JPEG"),
            "palette": variant["palette"],
            "difference_uri": it.to_data_uri(
                it.difference_map(original, image), fmt="PNG"
            ),
        }
    )
    st.session_state.result = selected
    st.session_state.artistic_influence = strength
    st.session_state.style_preset = next(
        (name for name, value in STYLE_PRESETS.items() if value == strength),
        "Custom",
    )


def comparison_difference_uri(result):
    if result.get("difference_uri"):
        return result["difference_uri"]
    original = it.image_from_data_uri(result["before_uri"])
    transformed = it.image_from_data_uri(result["after_uri"])
    return it.to_data_uri(it.difference_map(original, transformed), fmt="PNG")


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
    preset_col, strength_col, detail_col, action_col = st.columns(
        [0.95, 1.35, 1, 0.8], gap="large", vertical_alignment="bottom")
    with preset_col:
        st.selectbox(
            "Style preset",
            list(STYLE_PRESETS),
            index=2,
            key="style_preset",
            format_func=lambda value: (
                "Custom" if value == "Custom"
                else f"{value} ({STYLE_PRESETS[value]}%)"
            ),
            on_change=apply_style_preset,
        )
    with strength_col:
        strength = st.slider(
            "Artistic Influence", min_value=0, max_value=100, value=65, step=5,
            format="%d%%", key="artistic_influence",
            help="Lower values blend more original photo detail back in. 0% leaves it untouched; 100% applies the full style.",
            on_change=mark_custom_influence,
        )
        st.markdown(
            f'<p class="cm-help">{strength_note(strength)}</p>', unsafe_allow_html=True)
    with detail_col:
        detail = st.segmented_control(
            "Artistic Precision", list(DETAIL_LEVELS), default="Standard", key="detail",
            help="Standard matches the decoder's 128px training size. Sharper runs at 256px and is less tested.",
            on_change=clear_result,
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
if "history" not in st.session_state:
    st.session_state.history = []


def settings_key():
    return (
        hashlib.md5(content["data"]).hexdigest(),
        hashlib.md5(style["data"]).hexdigest(),
        strength,
        detail,
    )


if generate:
    status = None
    try:
        with st.status("Preparing your transformation...", expanded=False) as status:
            started = time.perf_counter()
            status.update(label="Preparing your images...", state="running")
            content_image = it.open_image(content["data"])
            style_image = it.open_image(style["data"])
            canvas = DETAIL_LEVELS[detail]
            content_tensor = preprocess_image(
                content_image, image_size=canvas, device=DEVICE)
            style_tensor = preprocess_image(
                style_image, image_size=canvas, device=DEVICE)
            status.update(label="Loading the trained decoder...", state="running")
            extractor, decoder = get_models(DEVICE, str(CHECKPOINT))
            status.update(label="Painting your image with the selected style...", state="running")
            output = transfer_style(
                content_tensor,
                style_tensor,
                alpha=1.0,
                feature_extractor=extractor,
                decoder=decoder,
                device=DEVICE,
            )
            status.update(label="Finishing the image and preparing downloads...", state="running")
            raw = denormalize_image(output)
            result_image, original_image = it.finish_result(
                raw, content_image, mix=strength / 100)
            status.update(label="Preparing your studio variations...", state="running")
            variants = []
            for variant_strength in VARIATION_STRENGTHS:
                variant_image = (
                    result_image
                    if variant_strength == strength
                    else it.finish_result(
                        raw, content_image, mix=variant_strength / 100
                    )[0]
                )
                variants.append(
                    {
                        "strength": variant_strength,
                        "png": it.encode(variant_image, "PNG"),
                        "palette": it.extract_palette(variant_image),
                    }
                )
            elapsed = time.perf_counter() - started
            result_record = {
                "key": settings_key(),
                "before_uri": it.to_data_uri(original_image, quality=92),
                "after_uri": it.to_data_uri(result_image, quality=92),
                "difference_uri": it.to_data_uri(
                    it.difference_map(original_image, result_image), fmt="PNG"
                ),
                "style_uri": it.to_data_uri(
                    it.fit_long_side(style_image, it.RESULT_MAX_LONG_SIDE),
                    quality=92,
                ),
                "png": it.encode(result_image, "PNG"),
                "jpg": it.encode(result_image, "JPEG"),
                "size": it.describe_size(result_image),
                "strength": strength,
                "detail": detail,
                "seconds": elapsed,
                "palette": it.extract_palette(result_image),
                "style_palette": style["palette"],
                "variants": variants,
                "history_view": False,
                "created_at": time.strftime("%H:%M:%S"),
            }
            status.update(label="Your transformation is ready!", state="complete", expanded=False)
        st.session_state.result = result_record
        history_record = {
            key: value
            for key, value in result_record.items()
            if key != "variants"
        }
        st.session_state.history = (
            [history_record] + st.session_state.history
        )[:MAX_HISTORY]
        st.toast("Your transformation is ready.")
    except Exception as exc:
        st.session_state.result = None
        if status is not None:
            status.update(label="Transformation could not be completed.", state="error", expanded=True)
        st.error(f"Could not generate the image: {exc}")

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------- result
result = st.session_state.result
if result is not None:
    difference_uri = comparison_difference_uri(result)
    if "difference_uri" not in result:
        result = {**result, "difference_uri": difference_uri}
        st.session_state.result = result
    st.markdown(
        '<div class="cm-result-head"><h2>A Tale of Two Perspectives</h2>'
        "<p>Explore the transformation from every angle</p></div>",
        unsafe_allow_html=True,
    )
    if not result.get("history_view") and ready and result["key"] != settings_key():
        st.markdown(
            '<div class="cm-stale">The images or settings have changed since this result. '
            "Select Bring Art to Life to refresh your creation</div>",
            unsafe_allow_html=True,
        )
    if result.get("history_view"):
        st.caption("Viewing a saved creation from this session.")
    view_col, side_col = st.columns([2.1, 1], gap="large")
    with view_col:
        mode_col, zoom_col, fullscreen_col = st.columns([1.35, 1, 0.8])
        with mode_col:
            compare_modes = ["Wipe", "Side-by-side", "Difference map"]
            if result.get("style_uri"):
                compare_modes.append("Muse Story")
            if st.session_state.get("compare_mode") not in compare_modes:
                st.session_state.compare_mode = "Wipe"
            compare_mode = st.segmented_control(
                "Compare view", compare_modes,
                default="Wipe", key="compare_mode",
            ) or "Wipe"
            if compare_mode == "Difference map":
                st.markdown(
                    parts.difference_map_note(), unsafe_allow_html=True
                )
                with st.popover("How to read this map", icon=":material/info:"):
                    st.markdown(
                        "The map compares aligned pixels in your photo and result, "
                        "then boosts their differences for visibility. "
                        "Brighter amber means a larger pixel change; dark navy means "
                        "little or no change.\n\n"
                        "It is a visual guide, not a quality score. Small shifts or "
                        "alignment differences can also create bright areas."
                    )
        with zoom_col:
            zoom = st.slider(
                "Zoom", min_value=100, max_value=200, value=100, step=10,
                format="%d%%", key="comparison_zoom",
            )
        with fullscreen_col:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            if st.button("Full screen", width="stretch", key="fullscreen_compare"):
                show_fullscreen_comparison(
                    result["before_uri"], result["after_uri"],
                    compare_mode, zoom, dark_mode, difference_uri,
                    result.get("style_uri"),
                )
        components.html(parts.compare_slider(
            result["before_uri"], result["after_uri"], dark_mode=dark_mode,
            mode=compare_mode, zoom=zoom,
            difference_uri=difference_uri, style_uri=result.get("style_uri")),
            height=480)
    with side_col:
        with st.container(key="resultside"):
            st.markdown(
                parts.result_facts(
                    result["size"], result["strength"], result["detail"], result["seconds"],
                    DEVICE_LABEL)
                + parts.swatches(result["style_palette"], "Style colours")
                + parts.swatches(result["palette"], "Result colours"),
                unsafe_allow_html=True,
            )
            st.markdown("#### Palette story")
            st.caption(it.palette_story(result["style_palette"], result["palette"]))
            st.download_button(
                "Download PNG", data=result["png"], file_name="colourmorph_result.png",
                mime="image/png", type="primary", width="stretch",
            )
            st.download_button(
                "Download JPG", data=result["jpg"], file_name="colourmorph_result.jpg",
                mime="image/jpeg", width="stretch",
            )
            showcase_png = build_showcase_png(
                result["before_uri"],
                result["png"],
                tuple(result["style_palette"]),
                result["strength"],
            )
            st.download_button(
                "Download showcase card",
                data=showcase_png,
                file_name="colourmorph_showcase.png",
                mime="image/png",
                width="stretch",
            )

if result is not None and result.get("variants"):
    st.markdown(
        '<div class="cm-result-head"><h2>Variation Lab</h2>'
        "<p>Compare three influence levels from one model render, then choose your favourite.</p></div>",
        unsafe_allow_html=True,
    )
    with st.popover("Preset vs. Variation Lab", icon=":material/help:"):
        st.markdown(
            "**They share the same influence scale, but work at different times.** "
            "A Style preset sets the starting influence before you generate. "
            "Variation Lab appears after generation and lets you compare the "
            "30%, 65%, and 100% versions. Choosing one updates the displayed result "
            "and its controls; it does not run a different style model."
        )
    with st.container(key="variation_lab"):
        variant_columns = st.columns(len(result["variants"]))
        for column, variant in zip(variant_columns, result["variants"]):
            with column:
                strength = variant["strength"]
                label = next(
                    (name for name, value in STYLE_PRESETS.items()
                     if value == strength),
                    f"{strength}% style",
                )
                st.image(
                    variant["png"],
                    caption=f"{label} · {strength}%",
                    width="stretch",
                )
                is_selected = strength == result["strength"]
                st.button(
                    "Selected" if is_selected else "Use this version",
                    key=f"variation_{strength}",
                    disabled=is_selected,
                    type="primary" if is_selected else "secondary",
                    width="stretch",
                    on_click=None if is_selected else select_variation,
                    args=(strength,),
                )

if st.session_state.history:
    st.markdown(
        '<div class="cm-result-head"><h2>Recent creations</h2>'
        "<p>Saved in this browser session only</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Open your session gallery", expanded=False):
        gallery_columns = st.columns(min(3, len(st.session_state.history)))
        for index, item in enumerate(st.session_state.history):
            with gallery_columns[index % len(gallery_columns)]:
                st.image(
                    item["png"],
                    caption=f'{item["strength"]}% influence · {item["detail"]} · {item["created_at"]}',
                    width="stretch",
                )
                if st.button(
                    "Compare this creation",
                    key=f"history_compare_{index}",
                    width="stretch",
                ):
                    show_history_result(index)
                    history_mode = st.session_state.get("compare_mode", "Wipe")
                    if history_mode == "Muse Story" and not item.get("style_uri"):
                        history_mode = "Wipe"
                    show_fullscreen_comparison(
                        item["before_uri"],
                        item["after_uri"],
                        history_mode,
                        st.session_state.get("comparison_zoom", 100),
                        dark_mode,
                        comparison_difference_uri(item),
                        item.get("style_uri"),
                    )
                st.download_button(
                    "Download this creation",
                    data=item["png"],
                    file_name=f"colourmorph_session_{index + 1}.png",
                    mime="image/png",
                    key=f"history_download_{index}",
                    width="stretch",
                )

# ---------------------------------------------------------------- creative process
animate_steps = not st.session_state.get("how_steps_seen", False)
st.session_state.how_steps_seen = True
st.markdown(parts.how_it_works(animate=animate_steps), unsafe_allow_html=True)
