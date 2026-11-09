# ui/

Everything that controls how ColourMorph looks. Nothing here touches the model.

| File | What it does | Edit it when you want to |
| --- | --- | --- |
| `theme.css` | Colours, fonts, spacing, responsive rules. The colours are variables at the top (`--cm-accent` and so on). | Change the look. |
| `parts.py` | Builds the HTML pieces: header, image frames, colour swatches, before/after slider. | Change the layout or wording of a piece. |
| `../image_tools.py` | Pillow-only helpers: safe image opening, preview sizing, palette extraction, result resizing. | Change how images are prepared or sized. |
| `../.streamlit/config.toml` | Streamlit theme colours and toolbar. | Change the base theme. |

`app.py` stays thin: it wires these together and calls the model exactly as before.
Run `python test_image_tools.py` to check the image helpers (no PyTorch needed).
