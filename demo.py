"""
demo.py — Gradio demo for face_id.

Run:
    python -m face_id.demo
"""

from typing import Optional, Tuple, Union

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from face_id import config
from face_id.gallery_manager import get_gallery, reload_gallery
from face_id.pipeline import enroll_from_array, verify_with_details


DEMO_CSS = """
.gradio-container { max-width: 1100px !important; }
.tabitem { padding-top: 0.5rem !important; }
"""


# ── Setup helpers ─────────────────────────────────────────────────────────────

def models_available() -> bool:
    return config.DETECTOR_MODEL.exists() and config.EMBEDDER_MODEL.exists()


def setup_instructions() -> str:
    return (
        "ONNX model weights are not installed.\n\n"
        f"Place these files in `face_id/models/`:\n"
        f"- `{config.DETECTOR_MODEL.name}`\n"
        f"- `{config.EMBEDDER_MODEL.name}`\n\n"
        "See `face_id/models/README.md` for download instructions."
    )


def _has_image(img: Union[Image.Image, np.ndarray, None]) -> bool:
    """True only when Gradio passed a real image (optional inputs may be blank arrays)."""
    if img is None:
        return False
    arr = np.array(img) if not isinstance(img, np.ndarray) else img
    if arr.size == 0 or arr.ndim < 2:
        return False
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return False
    # Gradio leaves uniform black/white placeholders in unused optional slots
    if float(arr.max() - arr.min()) < 5.0:
        return False
    return True


def pil_or_array_to_bgr(img: Union[Image.Image, np.ndarray, None]) -> Optional[np.ndarray]:
    if img is None:
        return None
    arr = np.array(img) if not isinstance(img, np.ndarray) else img
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def safe_enroll(name: str, image_bgr: np.ndarray) -> dict:
    try:
        return enroll_from_array(name, image_bgr)
    except FileNotFoundError as e:
        return {
            "enrolled": False,
            "name": name,
            "count": 0,
            "error": setup_instructions() + f"\n\n{e}",
        }


def safe_verify_with_details(image_bgr: np.ndarray) -> dict:
    try:
        return verify_with_details(image_bgr)
    except FileNotFoundError as e:
        return {
            "verified": False,
            "match": None,
            "confidence": 0.0,
            "all_scores": {},
            "face_found": False,
            "error": setup_instructions() + f"\n\n{e}",
            "bbox": None,
            "landmarks": None,
        }


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Draw verification result onto frame. Return BGR frame."""
    if not result.get("face_found"):
        cv2.putText(
            frame_bgr, "Searching...", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 160, 160), 2,
        )
        return frame_bgr

    color = (0, 220, 0) if result["verified"] else (0, 60, 220)
    label = result["match"] if result["verified"] else "Unknown"
    conf = result["confidence"]
    bbox = result.get("bbox")

    if bbox:
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
        text_y = min(y2 + 28, h - 10)
        cv2.putText(
            frame_bgr, f"{label}  {conf:.2f}", (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )
    else:
        h, w = frame_bgr.shape[:2]
        half = min(w, h) // 6
        cx, cy = w // 2, h // 2
        cv2.rectangle(
            frame_bgr,
            (cx - half, cy - half),
            (cx + half, cy + half),
            color, 3,
        )
        cv2.putText(
            frame_bgr, f"{label}  {conf:.2f}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2,
        )

    return frame_bgr


# ── Tab handlers ──────────────────────────────────────────────────────────────

def gallery_text() -> str:
    reload_gallery()
    gallery = get_gallery()
    identities = gallery.list_identities()
    if not identities:
        return "No identities enrolled."
    return "\n".join(
        f"• {name} — {gallery.count(name)} photo(s)"
        for name in identities
    )


def gallery_choices() -> list:
    reload_gallery()
    return get_gallery().list_identities()


def _enroll_status_message(name: str, count: int, already_existed: bool) -> str:
    noun = "image" if count == 1 else "images"
    if already_existed:
        return f"✓ Updated '{name}' ({count} reference {noun})"
    return f"✓ Enrolled '{name}' ({count} reference {noun})"


def do_enroll(
    name: str,
    photo1: Union[Image.Image, np.ndarray, None],
    photo2: Union[Image.Image, np.ndarray, None],
) -> Tuple[str, str]:
    if not models_available():
        return setup_instructions(), gallery_text()

    name = (name or "").strip()
    if not name:
        return "Enter an identity name.", gallery_text()

    if not _has_image(photo1):
        return "Add at least one photo.", gallery_text()

    reload_gallery()
    already_existed = name in get_gallery().list_identities()

    bgr1 = pil_or_array_to_bgr(photo1)
    result = safe_enroll(name, bgr1)
    if not result["enrolled"]:
        return result.get("error") or "No face found in the first photo.", gallery_text()

    count = result["count"]

    if _has_image(photo2):
        bgr2 = pil_or_array_to_bgr(photo2)
        result2 = safe_enroll(name, bgr2)
        if result2["enrolled"]:
            count = result2["count"]
        else:
            return (
                f"✓ First photo saved, but second failed: "
                f"{result2.get('error') or 'no face found'}.",
                gallery_text(),
            )

    return _enroll_status_message(name, count, already_existed), gallery_text()


def live_identify(
    frame: Union[Image.Image, np.ndarray, None],
) -> Tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, "Click record on the webcam to start."

    if not models_available():
        arr = pil_or_array_to_bgr(frame)
        rgb = bgr_to_rgb(arr) if arr is not None else None
        return rgb, setup_instructions()

    bgr = pil_or_array_to_bgr(frame)
    result = safe_verify_with_details(bgr)

    if result.get("error"):
        return bgr_to_rgb(bgr), result["error"]

    annotated = draw_result(bgr.copy(), result)
    rgb = bgr_to_rgb(annotated)

    if not result["face_found"]:
        status = "Searching..."
    elif result["verified"]:
        status = f"Match: {result['match']}  (confidence {result['confidence']:.2f})"
    else:
        status = f"Unknown  (best score {result['confidence']:.2f})"

    return rgb, status


def refresh_gallery() -> Tuple[str, gr.Dropdown]:
    return gallery_text(), gr.Dropdown(choices=gallery_choices(), value=None)


def remove_identity(name: str) -> Tuple[str, gr.Dropdown]:
    if not name:
        return refresh_gallery()
    gallery = get_gallery()
    gallery.remove(name)
    return refresh_gallery()


# ── App ───────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="face_id demo", css=DEMO_CSS) as demo:
        if models_available():
            gr.Markdown("### face_id demo — enroll a face, then identify live")
        else:
            gr.Markdown(f"### Setup required\n\n{setup_instructions()}")

        with gr.Tabs():
            # ── Tab 1: Enroll ─────────────────────────────────────────────
            with gr.Tab("Enroll"):
                with gr.Row():
                    enroll_name = gr.Textbox(
                        label="Identity name",
                        placeholder="e.g. Alice",
                        scale=3,
                    )
                    enroll_btn = gr.Button("Enroll", variant="primary", scale=1)

                with gr.Row():
                    photo1 = gr.Image(
                        label="Photo",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=240,
                    )
                    photo2 = gr.Image(
                        label="Second photo (optional)",
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=240,
                    )

                enroll_status = gr.Textbox(label="Status", interactive=False, lines=1)
                enroll_list = gr.Textbox(
                    label="Enrolled identities",
                    value=gallery_text(),
                    interactive=False,
                    lines=3,
                )
                enroll_btn.click(
                    do_enroll,
                    inputs=[enroll_name, photo1, photo2],
                    outputs=[enroll_status, enroll_list],
                )

            # ── Tab 2: Live Identify ────────────────────────────────────
            with gr.Tab("Live Identify"):
                gr.Markdown(
                    "Click **record** on the webcam — green = recognized, red = unknown."
                )
                with gr.Row():
                    webcam_in = gr.Image(
                        label="Webcam input",
                        sources=["webcam"],
                        type="numpy",
                        height=360,
                    )
                    webcam_out = gr.Image(
                        label="Annotated output",
                        streaming=True,
                        height=360,
                    )
                match_status = gr.Textbox(
                    label="Current match",
                    interactive=False,
                    lines=1,
                )
                webcam_in.stream(
                    live_identify,
                    inputs=[webcam_in],
                    outputs=[webcam_out, match_status],
                    time_limit=600,
                    stream_every=0.15,
                )

            # ── Tab 3: Gallery ────────────────────────────────────────────
            with gr.Tab("Gallery"):
                gallery_display = gr.Textbox(
                    label="Enrolled identities",
                    value=gallery_text(),
                    interactive=False,
                    lines=4,
                )
                with gr.Row():
                    identity_pick = gr.Dropdown(
                        label="Select identity to remove",
                        choices=gallery_choices(),
                    )
                    refresh_btn = gr.Button("Refresh")
                    remove_btn = gr.Button("Remove", variant="stop")

                refresh_btn.click(
                    refresh_gallery,
                    outputs=[gallery_display, identity_pick],
                )
                remove_btn.click(
                    remove_identity,
                    inputs=[identity_pick],
                    outputs=[gallery_display, identity_pick],
                )

    return demo


def main():
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
