"""Image helpers for the ColourMorph app. Pillow only, no PyTorch.

Everything here works on the user's pictures before and after the model runs:
reading uploads safely, making previews that share one frame, pulling out a
colour palette, and resizing the model output so it matches the content image.
"""
import base64
import binascii
import io

from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFilter, ImageFont, ImageOps

# A decompression-bomb guard. Pillow raises above 2x this value.
Image.MAX_IMAGE_PIXELS = 120_000_000

LANCZOS = Image.Resampling.LANCZOS

PREVIEW_LONG_SIDE = 1000   # previews are drawn at most this large
RESULT_MAX_LONG_SIDE = 1024
RESULT_MIN_LONG_SIDE = 384
DETAIL_RESTORE_AMOUNT = 1.0


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


def image_from_data_uri(data_uri: str) -> Image.Image:
    """Decode an image URI created by `to_data_uri`."""
    header, separator, payload = data_uri.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ImageReadError("This image preview could not be decoded.")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageReadError("This image preview could not be decoded.") from exc
    return open_image(data)


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


def palette_story(style_palette, result_palette, threshold: float = 90.0) -> str:
    """Summarize how many dominant result colours echo the style palette."""
    if not style_palette or not result_palette:
        return "There are not enough colours to compare these palettes yet."

    matches = []
    for style_colour in style_palette:
        style_rgb = tuple(
            int(style_colour[index:index + 2], 16) for index in (1, 3, 5)
        )
        closest = min(
            (
                _distance(
                    style_rgb,
                    tuple(int(colour[index:index + 2], 16) for index in (1, 3, 5)),
                ),
                colour,
            )
            for colour in result_palette
        )
        if closest[0] <= threshold:
            matches.append((closest[0], style_colour, closest[1]))

    if not matches:
        return (
            "The result has a distinct colour palette; its dominant colours "
            "do not closely match the artwork's top colours."
        )

    _, style_colour, result_colour = min(matches)
    return (
        f"The result echoes {len(matches)} of {len(style_palette)} signature "
        f"colours from the artwork. Closest echo: {style_colour} -> {result_colour}."
    )


def difference_map(original: Image.Image, result: Image.Image) -> Image.Image:
    """Highlight pixel changes with brighter amber tones."""
    original = original.convert("RGB")
    result = result.convert("RGB")
    if original.size != result.size:
        original = original.resize(result.size, LANCZOS)
    intensity = ImageOps.grayscale(ImageChops.difference(original, result))
    intensity = ImageEnhance.Contrast(intensity).enhance(2.5)
    return ImageOps.colorize(
        intensity, black="#171A2B", white="#FFD16A", blackpoint=0, whitepoint=210
    )


def _showcase_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def create_showcase_image(
    original: Image.Image,
    result: Image.Image,
    palette,
    strength: int,
) -> Image.Image:
    """Create a polished, shareable before-and-after presentation card."""
    width, height = 1400, 960
    margin, gap = 52, 24
    card_width = (width - 2 * margin - gap) // 2
    canvas = Image.new("RGB", (width, height), "#111321")
    draw = ImageDraw.Draw(canvas)
    title_font = _showcase_font(36)
    label_font = _showcase_font(22)
    detail_font = _showcase_font(18)

    draw.rounded_rectangle(
        (16, 16, width - 16, height - 16),
        radius=30,
        fill="#171A2C",
        outline="#363B5A",
        width=2,
    )
    draw.text((margin, 42), "COLOURMORPH  /  YOUR STYLE STORY",
              font=label_font, fill="#AEB8FF")
    draw.text((margin, 82), "A new perspective on your image",
              font=title_font, fill="#F7F7FC")
    draw.rounded_rectangle(
        (width - margin - 230, 48, width - margin, 104),
        radius=20,
        fill="#302C55",
    )
    draw.text((width - margin - 200, 66), f"{strength}% STYLE",
              font=label_font, fill="#F7F7FC")

    original = original.convert("RGB")
    result = result.convert("RGB")
    if original.size != result.size:
        original = original.resize(result.size, LANCZOS)

    card_y, card_height = 152, 520
    image_box = (card_width - 40, 408)
    cards = (
        (margin, original, "YOUR ORIGINAL"),
        (margin + card_width + gap, result, "STYLE TRANSFORMED"),
    )
    for x, image, label in cards:
        draw.rounded_rectangle(
            (x, card_y, x + card_width, card_y + card_height),
            radius=20,
            fill="#202438",
            outline="#343953",
            width=1,
        )
        draw.text((x + 20, card_y + 18), label,
                  font=label_font, fill="#D9DDF7")
        preview = ImageOps.contain(image, image_box, method=LANCZOS)
        preview_x = x + (card_width - preview.width) // 2
        preview_y = card_y + 66 + (image_box[1] - preview.height) // 2
        canvas.paste(preview, (preview_x, preview_y))

    draw.text((margin, 708), "PALETTE NOTES",
              font=label_font, fill="#AEB8FF")
    palette = list(palette or [])[:5]
    if palette:
        item_width = (width - 2 * margin) // len(palette)
        for index, colour in enumerate(palette):
            x = margin + index * item_width
            draw.rounded_rectangle(
                (x, 752, x + item_width - 10, 824),
                radius=14,
                fill="#202438",
            )
            draw.ellipse((x + 16, 773, x + 44, 801), fill=colour)
            draw.text((x + 56, 778), colour,
                      font=detail_font, fill="#F7F7FC")
    else:
        draw.text((margin, 760), "No palette data available.",
                  font=detail_font, fill="#D9DDF7")

    draw.text((margin, 875), "Created with ColourMorph",
              font=detail_font, fill="#A5AAC4")
    return canvas


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


def _restore_content_detail(
    result: Image.Image, content: Image.Image, amount: float
) -> Image.Image:
    """Restore a fixed amount of source luminance detail after style decoding."""
    result_ycbcr = result.convert("YCbCr")
    content_y = content.convert("YCbCr").getchannel("Y")
    low_frequency = content_y.filter(ImageFilter.GaussianBlur(radius=2))
    high_frequency = ImageChops.subtract(
        content_y, low_frequency, scale=1.0 / amount, offset=128
    )
    restored_y = ImageChops.add(
        result_ycbcr.getchannel("Y"), high_frequency, offset=-128
    )
    return Image.merge(
        "YCbCr",
        (restored_y, result_ycbcr.getchannel("Cb"), result_ycbcr.getchannel("Cr")),
    ).convert("RGB")


def finish_result(raw: Image.Image, content: Image.Image, mix: float = 1.0):
    """Turn the model's square output into the final picture.

    The model works on a square canvas (the content image is squeezed into it),
    so the output is stretched back to the content image's own proportions.
    A fixed high-pass residual restores luminance detail from the content image
    before the final blend. `mix` is how much of the model output to keep (0 to 1).
    The rest comes from the original photo, so mix=0 returns it untouched. The
    decoder is not a perfect copier: even with no style it redraws the photo softly.

    Returns (result, original) at the same size so they line up exactly in the
    before/after view.
    """
    size = result_size(content.size)
    result = raw.convert("RGB").resize(size, LANCZOS)
    original = content.convert("RGB").resize(size, LANCZOS)
    result = _restore_content_detail(result, original, DETAIL_RESTORE_AMOUNT)
    mix = min(1.0, max(0.0, float(mix)))
    if mix < 1.0:
        result = Image.blend(original, result, mix)
    return result, original
