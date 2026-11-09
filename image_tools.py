"""Image helpers for the ColourMorph app. Pillow only, no PyTorch.

Everything here works on the user's pictures before and after the model runs:
reading uploads safely, making previews that share one frame, pulling out a
colour palette, and resizing the model output so it matches the content image.
"""
import base64
import io

from PIL import Image, ImageFilter, ImageOps

# A decompression-bomb guard. Pillow raises above 2x this value.
Image.MAX_IMAGE_PIXELS = 120_000_000

LANCZOS = Image.Resampling.LANCZOS

PREVIEW_LONG_SIDE = 1000   # previews are drawn at most this large
RESULT_MAX_LONG_SIDE = 1024
RESULT_MIN_LONG_SIDE = 384


class ImageReadError(ValueError):
    """Raised with a message that is safe to show to the user."""


def open_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an upright RGB image.

    - Applies the EXIF rotation, so phone photos are not sideways.
    - Flattens transparency onto white, so transparent PNGs do not turn black.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError:
        raise ImageReadError(
            "This image is too large to open. Try one under about 100 megapixels.")
    except Exception:
        raise ImageReadError(
            "This file could not be read as an image. Use a JPG or PNG.")

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(backdrop, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def fit_long_side(image: Image.Image, long_side: int) -> Image.Image:
    """Shrink so the longer side is at most `long_side`. Keeps the aspect ratio
    and never enlarges. Uses Lanczos, which keeps edges crisp when shrinking."""
    width, height = image.size
    scale = long_side / max(width, height)
    if scale >= 1:
        return image.copy()
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), LANCZOS)


def to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(buffer, "JPEG",
                                  quality=quality, optimize=True, subsampling=0)
        mime = "image/jpeg"
    else:
        image.save(buffer, "PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def encode(image: Image.Image, fmt: str) -> bytes:
    """Full-quality bytes for the download buttons."""
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(buffer, "JPEG", quality=95,
                                  optimize=True, subsampling=0)
    else:
        image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def _distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def extract_palette(image: Image.Image, count: int = 5, min_gap: float = 42.0):
    """Dominant colours as hex strings, most common first.

    Colours that look almost the same are merged so the swatches are distinct.
    """
    small = fit_long_side(image, 128).convert("RGB")
    quantized = small.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    found = sorted(quantized.getcolors() or [], reverse=True)

    chosen = []
    for _, index in found:
        rgb = tuple(palette[index * 3:index * 3 + 3])
        if len(rgb) == 3 and all(_distance(rgb, other) >= min_gap for other in chosen):
            chosen.append(rgb)
        if len(chosen) == count:
            break
    return ["#%02X%02X%02X" % rgb for rgb in chosen]


def describe_size(image: Image.Image) -> str:
    width, height = image.size
    return f"{width:,} \u00d7 {height:,} px"


def result_size(content_size, long_side_cap: int = RESULT_MAX_LONG_SIDE):
    """Output size: the content image's aspect ratio, long side between
    RESULT_MIN_LONG_SIDE and the cap. Both sides are even numbers."""
    width, height = content_size
    long_side = min(
        max(max(width, height), RESULT_MIN_LONG_SIDE), long_side_cap)
    scale = long_side / max(width, height)
    out_w = max(2, round(width * scale / 2) * 2)
    out_h = max(2, round(height * scale / 2) * 2)
    return out_w, out_h


def finish_result(raw: Image.Image, content: Image.Image, mix: float = 1.0):
    """Turn the model's square output into the final picture.

    The model works on a square canvas (the content image is squeezed into it),
    so the output is stretched back to the content image's own proportions.
    A light sharpen offsets the softness of enlarging.

    `mix` is how much of the model output to keep (0 to 1). The rest comes from
    the original photo, so mix=0 returns the photo itself, untouched. The decoder
    is not a perfect copier: even with no style it redraws the photo softly.

    Returns (result, original) at the same size so they line up exactly in the
    before/after view.
    """
    size = result_size(content.size)
    result = raw.convert("RGB").resize(size, LANCZOS)
    if max(size) > max(raw.size):
        result = result.filter(ImageFilter.UnsharpMask(
            radius=1.4, percent=55, threshold=2))
    original = content.convert("RGB").resize(size, LANCZOS)
    mix = min(1.0, max(0.0, float(mix)))
    if mix < 1.0:
        result = Image.blend(original, result, mix)
    return result, original
