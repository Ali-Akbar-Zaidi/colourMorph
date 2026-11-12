"""HTML building blocks for the ColourMorph interface.

Each function returns a string. app.py decides where it goes on the page.
Text that comes from users (file names) is escaped before it is inserted.
"""
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent

MARK_SVG = (
    '<svg class="cm-mark" viewBox="0 0 32 32" aria-hidden="true">'
    '<circle cx="10" cy="10" r="7" fill="#3B4BDB"/>'
    '<circle cx="22" cy="10" r="7" fill="#E4577A" style="mix-blend-mode:multiply"/>'
    '<circle cx="10" cy="22" r="7" fill="#19A58A" style="mix-blend-mode:multiply"/>'
    '<circle cx="22" cy="22" r="7" fill="#F4B63F" style="mix-blend-mode:multiply"/>'
    "</svg>"
)

IMAGE_ICON = (
    '<svg viewBox="0 0 48 48" width="40" height="40" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="6" y="9" width="36" height="30" rx="5"/>'
    '<circle cx="17" cy="20" r="3.5"/>'
    '<path d="M6 34l11-9 8 6 7-7 10 9"/></svg>'
)


def load_css(dark_mode: bool = False) -> str:
    css = (HERE / "theme.css").read_text(encoding="utf-8")
    if not dark_mode:
        css = css.split("/* DARK_THEME_START */", maxsplit=1)[0]
    return css


def header(model_ready: bool, device_label: str) -> str:
    state = "ok" if model_ready else "warn"
    text = "Trained decoder found" if model_ready else "Decoder missing"
    return f"""
<header class="cm-header">
  <div class="cm-brand">{MARK_SVG}<span class="cm-wordmark">ColourMorph</span></div>
  <div class="cm-pills">
    <span class="cm-pill cm-pill-{state}"><i></i>{text}</span>
    <span class="cm-pill cm-pill-plain">Running on {html.escape(device_label)}</span>
  </div>
</header>
<section class="cm-hero">
  <div class="cm-hero-text">
    <p class="cm-eyebrow"><span></span> A LITTLE ART, A LOT OF YOU</p>
    <h1>Where Memories Meet Masterpieces</h1>
    <p>Let your photographs borrow the soul of timeless art through captivating colours, textures and brushstrokes</p>
  </div>
  <svg class="cm-art" viewBox="0 0 340 280" aria-hidden="true">
<circle cx="120" cy="110" r="95" fill="#3B4BDB" opacity=".92"/>
<circle cx="220" cy="110" r="95" fill="#E4577A" opacity=".92" style="mix-blend-mode:multiply"/>
<circle cx="120" cy="190" r="80" fill="#19A58A" opacity=".92" style="mix-blend-mode:multiply"/>
<circle cx="220" cy="190" r="80" fill="#F4B63F" opacity=".92" style="mix-blend-mode:multiply"/>
<circle cx="170" cy="150" r="126" fill="none" stroke="#3B4BDB" stroke-opacity=".16" stroke-width="1.5" stroke-dasharray="3 8"/>
</svg>
</section>
"""


def slot_heading(title: str, hint: str) -> str:
    return f'<div class="cm-slot-head"><h2>{html.escape(title)}</h2><p>{html.escape(hint)}</p></div>'


def swatches(colours, label: str) -> str:
    if not colours:
        return '<div class="cm-palette cm-palette-empty"></div>'
    items = "".join(
        f'<li title="{c}"><span style="background:{c}"></span><code>{c}</code></li>' for c in colours
    )
    return f'<div class="cm-palette"><p>{html.escape(label)}</p><ul>{items}</ul></div>'


def frame(data_uri: str, name: str, size_text: str, alt: str, tag: str = "") -> str:
    tag_html = f'<span class="cm-tag">{html.escape(tag)}</span>' if tag else ""
    return f"""
<figure class="cm-frame">
  <div class="cm-stage">
    <img class="cm-bg" src="{data_uri}" alt="">
    <img class="cm-fg" src="{data_uri}" alt="{html.escape(alt)}">
    {tag_html}
  </div>
  <figcaption><span class="cm-name" title="{html.escape(name)}">{html.escape(name)}</span><span class="cm-dims">{html.escape(size_text)}</span></figcaption>
</figure>
"""


def empty_frame(message: str) -> str:
    return f"""
<figure class="cm-frame cm-frame-empty">
  <div class="cm-stage">
    <div class="cm-empty">{IMAGE_ICON}<span>{html.escape(message)}</span></div>
  </div>
  <figcaption><span class="cm-name">No image yet</span><span class="cm-dims"></span></figcaption>
</figure>
"""


def error_frame(message: str) -> str:
    return f"""
<figure class="cm-frame cm-frame-error">
  <div class="cm-stage"><div class="cm-empty"><span>{html.escape(message)}</span></div></div>
  <figcaption><span class="cm-name">Could not use this file</span><span class="cm-dims"></span></figcaption>
</figure>
"""


def result_facts(size_text: str, strength: int, detail_label: str, seconds: float, device_label: str) -> str:
    rows = [
        ("Size", size_text),
        ("Style applied", f"{strength}%"),
        ("Detail", detail_label),
        ("Time", f"{seconds:.1f} s on {device_label}"),
    ]
    body = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in rows)
    return f'<dl class="cm-facts">{body}</dl>'


def difference_map_note() -> str:
    return """
<div class="cm-map-note" role="note">
  <span class="cm-map-note-mark" aria-hidden="true">✦</span>
  <div>
    <strong>See where the image changed</strong>
    <p>Brighter amber areas indicate greater differences between the images, while dark navy areas indicate little or no difference.</p>
  </div>
</div>
"""


def how_it_works(animate: bool = False) -> str:
    steps = [
        (
            "01",
            "Set the Scene",
            "Choose the photograph that tells your story",
            '<rect x="4" y="5" width="11" height="14" rx="2"/><path d="m6 16 3-4 2 2 2-3 2 3"/><rect x="15" y="9" width="5" height="10" rx="1"/><path d="M17 12h1m-1 3h1"/>',
        ),
        (
            "02",
            "Find Your Muse",
            "Select an artwork that speaks to your imagination",
            '<path d="M4 5h16v14H4z"/><path d="M8 9h8M8 12h5M8 15h7"/><circle cx="17" cy="16" r="3"/>',
        ),
        (
            "03",
            "Capture the Essence",
            "Extract the colours, patterns and textures that define its beauty",
            '<path d="M5 7h14M5 12h14M5 17h14"/><circle cx="10" cy="7" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="8" cy="17" r="2"/>',
        ),
        (
            "04",
            "Reimagine Your Image",
            "Blend artistic inspiration into your original photograph",
            '<path d="m7 17 10-10 3 3-10 10-4 1z"/><path d="m15 9 3 3M5 5h5M4 9v2"/>',
        ),
        (
            "05",
            "Own Your Masterpiece",
            "Explore the transformation and save your creation",
            '<path d="M4 5h16v14H4z"/><path d="M12 5v14"/><path d="m8 10-2 2 2 2m8-4 2 2-2 2"/><path d="M9 22h6"/>',
        ),
    ]
    cards = "".join(
        f"""
<article class="cm-step">
  <div class="cm-step-top"><span class="cm-step-number">{number}</span><span class="cm-step-label">STEP</span></div>
  <div class="cm-step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{icon}</svg></div>
  <h3>{title}</h3>
  <p>{description}</p>
</article>"""
        for number, title, description, icon in steps
    )
    animation_class = " cm-steps-enter" if animate else ""
    return f"""
<section class="cm-how{animation_class}" aria-labelledby="cm-how-title">
  <div class="cm-how-heading">
    <p class="cm-how-kicker"><span></span> THE ART BEHIND THE TRANSFORMATION</p>
    <h2 id="cm-how-title">Two Worlds, <span>One Creation</span></h2>
    <p class="cm-how-intro">Where your memories meet the imagination of great artists</p>
  </div>
  <div class="cm-step-grid">{cards}</div>
  <p class="cm-how-note"><span>✦</span> Every image holds a story, every artwork holds a feeling, ColourMorph brings them together</p>
</section>
"""


def _side_by_side_comparison(
    before_uri: str,
    after_uri: str,
    dark_mode: bool,
    zoom: int,
) -> str:
    theme = (
        "--cmp-bg:#191b28;--cmp-ink:#f2f1f8;--cmp-outline:rgba(225,228,255,.2)"
        if dark_mode
        else "--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12)"
    )
    stage = """
<div class="stage side-stage">
  <figure><img class="compare-image" src="__BEFORE__" alt="Original image">
    <figcaption>Original</figcaption></figure>
  <figure><img class="compare-image" src="__AFTER__" alt="Stylized result">
    <figcaption>ColourMorph</figcaption></figure>
</div>
"""

    document = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12);--zoom:__ZOOM__;__THEME__}
*{box-sizing:border-box;margin:0}
html,body{height:100%;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viewer{position:relative;height:100%;min-height:300px;overflow:hidden;border-radius:18px;
  background:var(--cmp-bg);outline:1px solid var(--cmp-outline);animation:arrive .55s cubic-bezier(.2,.75,.25,1) both}
.stage{position:absolute;inset:0;overflow:hidden}
.compare-image{width:100%;height:100%;object-fit:contain;display:block;transform:scale(var(--zoom));
  transition:transform .28s ease,opacity .28s ease}
.side-stage{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--cmp-outline)}
.side-stage figure{min-width:0;height:100%;display:flex;flex-direction:column;background:var(--cmp-bg);overflow:hidden}
.side-stage img{height:calc(100% - 34px)}
.side-stage figcaption{height:34px;display:grid;place-items:center;color:var(--cmp-ink);font-size:13px;font-weight:600}
@keyframes arrive{from{opacity:0;transform:translateY(8px) scale(.985)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.viewer,.compare-image{animation:none!important;transition:none!important}}
</style></head><body>
<div class="viewer" role="region" aria-label="Side-by-side image comparison">
__STAGE__
 </div></body></html>"""
    return (
        document.replace("__STAGE__", stage)
        .replace("__THEME__", theme)
        .replace("__ZOOM__", f"{max(100, min(200, int(zoom))) / 100:.2f}")
        .replace("__BEFORE__", before_uri)
        .replace("__AFTER__", after_uri)
    )


def _difference_comparison(
    difference_uri: str,
    dark_mode: bool,
    zoom: int,
) -> str:
    theme = (
        "--cmp-bg:#191b28;--cmp-ink:#f2f1f8;--cmp-outline:rgba(225,228,255,.2)"
        if dark_mode
        else "--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12)"
    )
    document = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12);--zoom:__ZOOM__;__THEME__}
*{box-sizing:border-box;margin:0}
html,body{height:100%;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viewer{position:relative;height:100%;min-height:300px;overflow:hidden;border-radius:18px;
  background:var(--cmp-bg);outline:1px solid var(--cmp-outline);animation:arrive .55s cubic-bezier(.2,.75,.25,1) both}
figure{position:absolute;inset:0;display:flex;flex-direction:column;overflow:hidden}
.difference-image{width:100%;height:calc(100% - 38px);object-fit:contain;display:block;transform:scale(var(--zoom));
  transition:transform .28s ease}
figcaption{height:38px;display:grid;place-items:center;color:var(--cmp-ink);font-size:13px;font-weight:600}
@keyframes arrive{from{opacity:0;transform:translateY(8px) scale(.985)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.viewer,.difference-image{animation:none!important;transition:none!important}}
</style></head><body>
<div class="viewer" role="region" aria-label="Difference map image comparison">
  <figure><img class="difference-image" src="__DIFFERENCE__" alt="Map of pixel changes">
    <figcaption>Brighter areas indicate greater differences between the images</figcaption></figure>
</div></body></html>"""
    return (
        document.replace("__THEME__", theme)
        .replace("__ZOOM__", f"{max(100, min(200, int(zoom))) / 100:.2f}")
        .replace("__DIFFERENCE__", difference_uri)
    )


def _muse_story_comparison(
    before_uri: str,
    style_uri: str,
    after_uri: str,
    dark_mode: bool,
    zoom: int,
) -> str:
    theme = (
        "--cmp-bg:#191b28;--cmp-ink:#f2f1f8;--cmp-outline:rgba(225,228,255,.2)"
        if dark_mode
        else "--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12)"
    )
    stage = """
<div class="stage story-stage">
  <figure><img class="compare-image" src="__BEFORE__" alt="Original photo">
    <figcaption>Original</figcaption></figure>
  <figure><img class="compare-image" src="__STYLE__" alt="Artistic muse">
    <figcaption>Muse</figcaption></figure>
  <figure><img class="compare-image" src="__AFTER__" alt="ColourMorph result">
    <figcaption>ColourMorph</figcaption></figure>
</div>
"""
    document = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12);--zoom:__ZOOM__;__THEME__}
*{box-sizing:border-box;margin:0}
html,body{height:100%;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viewer{position:relative;height:100%;min-height:300px;overflow:hidden;border-radius:18px;
  background:var(--cmp-bg);outline:1px solid var(--cmp-outline);animation:arrive .55s cubic-bezier(.2,.75,.25,1) both}
.stage{position:absolute;inset:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:1px;background:var(--cmp-outline)}
figure{min-width:0;height:100%;display:flex;flex-direction:column;background:var(--cmp-bg);overflow:hidden}
.compare-image{width:100%;height:calc(100% - 34px);object-fit:contain;display:block;transform:scale(var(--zoom));
  transition:transform .28s ease}
figcaption{height:34px;display:grid;place-items:center;color:var(--cmp-ink);font-size:13px;font-weight:600}
@keyframes arrive{from{opacity:0;transform:translateY(8px) scale(.985)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.viewer,.compare-image{animation:none!important;transition:none!important}}
</style></head><body>
<div class="viewer" role="region" aria-label="Muse story image comparison">
__STAGE__
</div></body></html>"""
    return (
        document.replace("__STAGE__", stage)
        .replace("__THEME__", theme)
        .replace("__ZOOM__", f"{max(100, min(200, int(zoom))) / 100:.2f}")
        .replace("__BEFORE__", before_uri)
        .replace("__STYLE__", style_uri)
        .replace("__AFTER__", after_uri)
    )


def compare_slider(
    before_uri: str,
    after_uri: str,
    height: int = 520,
    dark_mode: bool = False,
    mode: str = "Wipe",
    zoom: int = 100,
    difference_uri: str | None = None,
    style_uri: str | None = None,
) -> str:
    """Build a self-contained, accessible image comparison for a Streamlit iframe."""
    theme = (
        "--cmp-bg:#191b28;--cmp-ink:#f2f1f8;--cmp-outline:rgba(225,228,255,.2);"
        "--cmp-knob:#292c3d;--cmp-chip:rgba(13,14,22,.8)"
        if dark_mode
        else "--cmp-bg:#E8ECF7;--cmp-ink:#14183A;--cmp-outline:rgba(20,24,58,.12);"
        "--cmp-knob:#fff;--cmp-chip:rgba(20,24,58,.72)"
    )
    if mode == "Side-by-side":
        return _side_by_side_comparison(before_uri, after_uri, dark_mode, zoom)
    if mode == "Difference map":
        if difference_uri is None:
            raise ValueError("A difference image is required for Difference map mode.")
        return _difference_comparison(difference_uri, dark_mode, zoom)
    if mode == "Muse Story":
        if style_uri is None:
            raise ValueError("A style image is required for Muse Story mode.")
        return _muse_story_comparison(
            before_uri, style_uri, after_uri, dark_mode, zoom
        )
    if mode != "Wipe":
        mode = "Wipe"
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--pos:50%;--accent:#3B4BDB;--zoom:__ZOOM__;__THEME__}
*{box-sizing:border-box;margin:0}
html,body{height:100%;background:transparent;font-family:"Hanken Grotesk",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#cmp{position:relative;height:100%;border-radius:18px;overflow:hidden;background:var(--cmp-bg);
  touch-action:pan-y;user-select:none;-webkit-user-select:none;cursor:ew-resize;
  outline:1px solid var(--cmp-outline);outline-offset:-1px}
.layer{position:absolute;inset:0}
.layer img{width:100%;height:100%;object-fit:contain;display:block;-webkit-user-drag:none;
  transform:scale(var(--zoom));transition:transform .28s ease}
#before{clip-path:inset(0 calc(100% - var(--pos)) 0 0)}
#line{position:absolute;top:0;bottom:0;left:var(--pos);width:2px;margin-left:-1px;background:#fff;
  box-shadow:0 0 0 1px var(--cmp-outline)}
#knob{position:absolute;top:50%;left:var(--pos);width:46px;height:46px;margin:-23px 0 0 -23px;border-radius:50%;
  background:var(--cmp-knob);color:var(--cmp-ink);display:grid;place-items:center;box-shadow:0 2px 10px rgba(0,0,0,.35);
  transition:box-shadow .15s}
#cmp:focus-visible{outline:3px solid var(--accent);outline-offset:-3px}
 #cmp:focus-visible #knob,#cmp.drag #knob{box-shadow:0 0 0 4px rgba(59,75,219,.45),0 2px 10px rgba(0,0,0,.35)}
.chip{position:absolute;bottom:14px;padding:6px 12px;border-radius:999px;font-size:13px;font-weight:600;
  color:#fff;background:var(--cmp-chip);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);pointer-events:none}
#c1{left:14px}#c2{right:14px}
</style></head><body>
<div id="cmp" tabindex="0" role="slider" aria-label="Compare original and result"
     aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
  <div class="layer"><img alt="Result" src="__AFTER__"></div>
  <div class="layer" id="before"><img alt="Original" src="__BEFORE__"></div>
  <div id="line"></div>
  <div id="knob" aria-hidden="true">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
         stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6"/></svg>
  </div>
  <span class="chip" id="c1">Original</span><span class="chip" id="c2">ColourMorph</span>
</div>
<script>
(function(){
  var el=document.getElementById('cmp'),pos=100,raf=null,dragging=false;
  function set(p){pos=Math.max(0,Math.min(100,p));el.style.setProperty('--pos',pos+'%');
    el.setAttribute('aria-valuenow',Math.round(pos));}
  function fromEvent(e){var r=el.getBoundingClientRect();set((e.clientX-r.left)/r.width*100);}
  function stop(){if(raf){cancelAnimationFrame(raf);raf=null;}}
  el.addEventListener('pointerdown',function(e){stop();dragging=true;el.classList.add('drag');
    try{el.setPointerCapture(e.pointerId);}catch(_){ }fromEvent(e);});
  el.addEventListener('pointermove',function(e){if(dragging)fromEvent(e);});
  function up(){dragging=false;el.classList.remove('drag');}
  el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up);
  el.addEventListener('keydown',function(e){
    var k=e.key,step=e.shiftKey?10:3;
    if(k==='ArrowLeft'||k==='ArrowDown'){stop();set(pos-step);e.preventDefault();}
    else if(k==='ArrowRight'||k==='ArrowUp'){stop();set(pos+step);e.preventDefault();}
    else if(k==='Home'){stop();set(0);e.preventDefault();}
    else if(k==='End'){stop();set(100);e.preventDefault();}});
  /* One reveal when the result first appears: the original sweeps away. */
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(reduce){set(50);return;}
  set(100);var t0=null,from=100,to=50,dur=1100;
  function ease(t){return 1-Math.pow(1-t,3);}
  function tick(ts){if(t0===null)t0=ts;var t=Math.min(1,(ts-t0)/dur);set(from+(to-from)*ease(t));
    if(t<1)raf=requestAnimationFrame(tick);else raf=null;}
  setTimeout(function(){raf=requestAnimationFrame(tick);},250);
})();
</script></body></html>""".replace("__BEFORE__", before_uri).replace("__AFTER__", after_uri).replace(
        "__THEME__", theme
    ).replace("__ZOOM__", f"{max(100, min(200, int(zoom))) / 100:.2f}")
