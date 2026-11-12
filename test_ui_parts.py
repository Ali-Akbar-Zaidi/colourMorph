"""Tests for the comparison view markup.

Run with: python test_ui_parts.py
"""
from ui.parts import compare_slider, difference_map_note, load_css


def test_comparison_view_uses_clear_images_on_solid_background():
    markup = compare_slider("data:image/png;base64,before", "data:image/png;base64,after")

    assert '<img alt="Result" src="data:image/png;base64,after">' in markup
    assert '<img alt="Original" src="data:image/png;base64,before">' in markup
    assert "background:var(--cmp-bg)" in markup
    assert 'class="bg"' not in markup
    assert "filter:blur(28px)" not in markup
    assert 'role="slider"' in markup
    assert "prefers-reduced-motion" in markup


def test_removed_fade_mode_falls_back_to_wipe():
    markup = compare_slider(
        "data:image/png;base64,before",
        "data:image/png;base64,after",
        mode="Fade",
        zoom=140,
    )

    assert 'aria-label="Compare original and result"' in markup
    assert 'id="cmp"' in markup
    assert 'id="fade-range"' not in markup
    assert 'aria-label="Original image visibility"' not in markup


def test_side_by_side_comparison_labels_both_images():
    markup = compare_slider(
        "data:image/png;base64,before",
        "data:image/png;base64,after",
        mode="Side-by-side",
    )

    assert "<figcaption>Original</figcaption>" in markup
    assert "<figcaption>ColourMorph</figcaption>" in markup
    assert 'aria-label="Side-by-side image comparison"' in markup
    assert "prefers-reduced-motion" in markup
    assert 'src="data:image/png;base64,after"' in markup
    assert 'src="data:image/png;base64,before"' in markup
    assert "__AFTER__" not in markup
    assert "__BEFORE__" not in markup


def test_difference_map_displays_the_generated_change_image():
    markup = compare_slider(
        "data:image/png;base64,before",
        "data:image/png;base64,after",
        mode="Difference map",
        difference_uri="data:image/png;base64,difference",
    )

    assert 'aria-label="Difference map image comparison"' in markup
    assert 'alt="Map of pixel changes"' in markup
    assert 'src="data:image/png;base64,difference"' in markup
    assert "Brighter areas indicate greater differences between the images" in markup


def test_difference_map_note_explains_the_visual_encoding():
    markup = difference_map_note()

    assert 'class="cm-map-note"' in markup
    assert "Brighter amber areas indicate greater differences between the images" in markup
    assert "dark navy areas indicate little or no difference" in markup


def test_dark_theme_styles_streamlit_text_and_popovers():
    light_css = load_css()
    dark_css = load_css(dark_mode=True)

    assert "[data-testid=\"stPopoverBody\"]" not in light_css
    assert "[data-testid=\"stPopoverBody\"]" in dark_css
    assert "[data-testid=\"stMarkdownContainer\"] p" in dark_css
    assert "--cm-ink-soft: #D0D3E5;" in dark_css
    assert '[data-testid="stApp"] button[data-testid="stBaseButton-primary"]' in dark_css
    assert '[data-testid="stApp"] button[data-variant="segmented_control"]' in dark_css
    assert "background-color: #202232 !important;" in dark_css


def test_difference_map_requires_its_image():
    try:
        compare_slider(
            "data:image/png;base64,before",
            "data:image/png;base64,after",
            mode="Difference map",
        )
    except ValueError as exc:
        assert "difference image is required" in str(exc)
    else:
        raise AssertionError("expected Difference map mode to require an image")


def test_muse_story_comparison_shows_all_three_images():
    markup = compare_slider(
        "data:image/png;base64,before",
        "data:image/png;base64,after",
        mode="Muse Story",
        style_uri="data:image/png;base64,muse",
    )

    assert 'aria-label="Muse story image comparison"' in markup
    assert 'alt="Original photo"' in markup
    assert 'src="data:image/png;base64,muse"' in markup
    assert "<figcaption>Muse</figcaption>" in markup
    assert "__STYLE__" not in markup


if __name__ == "__main__":
    test_comparison_view_uses_clear_images_on_solid_background()
    test_removed_fade_mode_falls_back_to_wipe()
    test_side_by_side_comparison_labels_both_images()
    test_difference_map_displays_the_generated_change_image()
    test_difference_map_note_explains_the_visual_encoding()
    test_dark_theme_styles_streamlit_text_and_popovers()
    test_difference_map_requires_its_image()
    test_muse_story_comparison_shows_all_three_images()
    print("passed test_comparison_view_uses_clear_images_on_solid_background")
    print("passed test_removed_fade_mode_falls_back_to_wipe")
    print("passed test_side_by_side_comparison_labels_both_images")
    print("passed test_difference_map_displays_the_generated_change_image")
    print("passed test_difference_map_note_explains_the_visual_encoding")
    print("passed test_dark_theme_styles_streamlit_text_and_popovers")
    print("passed test_difference_map_requires_its_image")
    print("passed test_muse_story_comparison_shows_all_three_images")
    print("all tests passed")
