"""Tests for image_tools.py. They need Pillow only, not PyTorch.

Run with: python test_image_tools.py
"""
import io

from PIL import Image

import image_tools as it


def _png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_transparent_png_becomes_white_not_black():
    data = _png_bytes(Image.new("RGBA", (20, 20), (0, 0, 0, 0)))
    assert it.open_image(data).getpixel((5, 5)) == (255, 255, 255)


def test_exif_rotation_is_applied():
    image = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees when displayed
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    assert it.open_image(buffer.getvalue()).size == (20, 40)


def test_unreadable_file_gives_a_friendly_error():
    try:
        it.open_image(b"definitely not an image")
    except it.ImageReadError as exc:
        assert "could not be read" in str(exc)
    else:
        raise AssertionError("expected ImageReadError")


def test_fit_long_side_keeps_ratio_and_never_enlarges():
    big = Image.new("RGB", (4000, 1000))
    assert it.fit_long_side(big, 1000).size == (1000, 250)
    small = Image.new("RGB", (300, 200))
    assert it.fit_long_side(small, 1000).size == (300, 200)


def test_result_size_follows_content_ratio_with_even_sides():
    for content in [(1280, 853), (4000, 3000), (200, 100), (1000, 3000), (777, 555)]:
        width, height = it.result_size(content)
        assert width % 2 == 0 and height % 2 == 0
        assert max(it.RESULT_MIN_LONG_SIDE, 1) <= max(
            width, height) <= it.RESULT_MAX_LONG_SIDE
        assert abs(width / height - content[0] / content[1]) < 0.01


def test_finish_result_matches_content_proportions():
    content = Image.new("RGB", (1600, 900), (10, 120, 200))
    raw = Image.new("RGB", (128, 128), (200, 120, 10))
    result, original = it.finish_result(raw, content)
    assert result.size == original.size
    assert abs(result.size[0] / result.size[1] - 16 / 9) < 0.01


def test_finish_result_restores_content_edges():
    content = Image.new("RGB", (384, 384), (0, 0, 0))
    content.paste((255, 255, 255), (192, 0, 384, 384))
    raw = Image.new("RGB", (128, 128), (128, 128, 128))

    result, _ = it.finish_result(raw, content, mix=1)

    assert result.getpixel((190, 192))[0] < result.getpixel((194, 192))[0]


def test_zero_mix_returns_the_original_photo():
    content = Image.new("RGB", (800, 600), (10, 120, 200))
    raw = Image.new("RGB", (128, 128), (200, 120, 10))
    result, original = it.finish_result(raw, content, mix=0)
    assert result.tobytes() == original.tobytes()
    full, _ = it.finish_result(raw, content, mix=1)
    half, _ = it.finish_result(raw, content, mix=0.5)
    assert full.getpixel((10, 10)) != half.getpixel(
        (10, 10)) != result.getpixel((10, 10))


def test_palette_finds_distinct_colours():
    image = Image.new("RGB", (100, 100), (220, 40, 40))
    image.paste((40, 60, 220), (0, 0, 100, 50))
    palette = it.extract_palette(image, count=5)
    assert 2 <= len(palette) <= 5
    assert len(set(palette)) == len(palette)
    assert all(len(colour) == 7 and colour.startswith("#")
               for colour in palette)


def test_image_data_uri_round_trips():
    image = Image.new("RGB", (18, 12), (90, 120, 210))

    decoded = it.image_from_data_uri(it.to_data_uri(image, fmt="PNG"))

    assert decoded.size == image.size
    assert decoded.getpixel((4, 4)) == image.getpixel((4, 4))


def test_palette_story_reports_shared_signature_colours():
    story = it.palette_story(
        ["#112233", "#DDEEFF"],
        ["#112233", "#DDEEFF"],
    )

    assert "echoes 2 of 2 signature colours" in story
    assert "#112233 -> #112233" in story


def test_difference_map_highlights_changed_pixels():
    original = Image.new("RGB", (20, 20), (20, 30, 40))
    result = original.copy()
    result.paste((230, 220, 10), (8, 8, 12, 12))

    difference = it.difference_map(original, result)

    assert difference.size == original.size
    assert difference.getpixel((10, 10)) != difference.getpixel((2, 2))


def test_showcase_export_contains_both_images_and_palette():
    original = Image.new("RGB", (40, 30), (20, 30, 40))
    result = Image.new("RGB", (40, 30), (230, 220, 10))

    showcase = it.create_showcase_image(
        original, result, ["#E6DC0A", "#14202A"], strength=65
    )

    assert showcase.size == (1400, 960)
    assert showcase.mode == "RGB"
    assert showcase.getpixel((600, 300)) != showcase.getpixel((900, 300))


TESTS = [value for name, value in sorted(
    globals().items()) if name.startswith("test_")]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"passed {test.__name__}")
    print("all tests passed")
