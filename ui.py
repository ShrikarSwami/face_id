"""
ui.py — local Tkinter app to test the face_id library.

Run:
    python -m face_id.ui
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np
from PIL import Image, ImageTk

from face_id import config
from face_id.pipeline import enroll, verify, enroll_from_array, verify_from_array
from face_id.gallery_manager import get_gallery, reload_gallery
from face_id.verify import _draw_overlay


# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_OK = "#22c55e"
COLOR_FAIL = "#ef4444"
COLOR_NEUTRAL = "#9ca3af"
COLOR_BG = "#1e1e2e"
COLOR_PANEL = "#2a2a3c"
COLOR_TEXT = "#e4e4ef"
COLOR_ACCENT = "#7c9eff"


def models_available() -> bool:
    return config.DETECTOR_MODEL.exists() and config.EMBEDDER_MODEL.exists()


def model_missing_message(exc: Optional[Exception] = None) -> str:
    base = (
        "ONNX model weights not found.\n\n"
        f"Expected:\n  {config.DETECTOR_MODEL}\n  {config.EMBEDDER_MODEL}\n\n"
        "Download weights per face_id/models/README.md "
        "(InsightFace buffalo_sc pack: det_500m.onnx + w600k_mbf.onnx)."
    )
    if exc:
        return f"{base}\n\nDetails: {exc}"
    return base


def safe_enroll(name: str, image) -> dict:
    try:
        return enroll(name, image)
    except FileNotFoundError as e:
        return {
            "enrolled": False,
            "name": name,
            "count": 0,
            "error": model_missing_message(e),
        }


def safe_verify(image, threshold: float = None) -> dict:
    try:
        return verify(image, threshold=threshold)
    except FileNotFoundError as e:
        return {
            "verified": False,
            "match": None,
            "confidence": 0.0,
            "all_scores": {},
            "face_found": False,
            "error": model_missing_message(e),
        }


def bgr_to_photo(bgr: np.ndarray, max_size: tuple = (480, 360)) -> ImageTk.PhotoImage:
    """Resize BGR frame for Tkinter display."""
    h, w = bgr.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def capture_webcam_frame() -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── App ───────────────────────────────────────────────────────────────────────

class FaceIdApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("face_id — verification UI")
        self.geometry("900x720")
        self.minsize(800, 600)
        self.configure(bg=COLOR_BG)

        self._enroll_image: Optional[np.ndarray] = None
        self._verify_image: Optional[np.ndarray] = None
        self._enroll_photo: Optional[ImageTk.PhotoImage] = None
        self._verify_photo: Optional[ImageTk.PhotoImage] = None
        self._live_photo: Optional[ImageTk.PhotoImage] = None

        self._live_active = False
        self._live_cap: Optional[cv2.VideoCapture] = None
        self._live_job: Optional[str] = None
        self._fps_frames = 0
        self._fps_timer = time.time()
        self._display_fps = 0.0
        self._app_icon: Optional[ImageTk.PhotoImage] = None

        self._set_app_icon()
        self._build_header()
        self._build_notebook()
        self._refresh_identity_list()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _set_app_icon(self):
        if not config.APP_ICON.exists():
            return
        try:
            icon_img = Image.open(config.APP_ICON)
            self._app_icon = ImageTk.PhotoImage(icon_img)
            self.iconphoto(True, self._app_icon)
        except OSError:
            pass

    def _build_header(self):
        header = tk.Frame(self, bg=COLOR_PANEL, pady=10, padx=16)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="face_id",
            font=("Helvetica", 18, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_PANEL,
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="few-shot face verification",
            font=("Helvetica", 11),
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
        ).pack(side=tk.LEFT, padx=(12, 0))

        if not models_available():
            tk.Label(
                header,
                text="⚠ Models missing — see models/README.md",
                font=("Helvetica", 10),
                fg="#fbbf24",
                bg=COLOR_PANEL,
            ).pack(side=tk.RIGHT)

    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=[14, 8], font=("Helvetica", 11))
        style.map("TNotebook.Tab", background=[("selected", COLOR_PANEL)])

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        enroll_frame = tk.Frame(notebook, bg=COLOR_BG)
        verify_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(enroll_frame, text="  Enroll  ")
        notebook.add(verify_frame, text="  Verify  ")

        self._build_enroll_tab(enroll_frame)
        self._build_verify_tab(verify_frame)

    def _build_enroll_tab(self, parent: tk.Frame):
        left = tk.Frame(parent, bg=COLOR_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right = tk.Frame(parent, bg=COLOR_PANEL, width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        # Name input
        tk.Label(left, text="Identity name", fg=COLOR_TEXT, bg=COLOR_BG,
                 font=("Helvetica", 11)).pack(anchor=tk.W)
        self._enroll_name = tk.Entry(
            left, font=("Helvetica", 13), bg=COLOR_PANEL, fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT, relief=tk.FLAT,
        )
        self._enroll_name.pack(fill=tk.X, pady=(4, 12), ipady=6)

        # Buttons
        btn_row = tk.Frame(left, bg=COLOR_BG)
        btn_row.pack(fill=tk.X, pady=(0, 12))

        tk.Button(
            btn_row, text="Upload photo", command=self._enroll_upload,
            bg=COLOR_PANEL, fg=COLOR_TEXT, relief=tk.FLAT, padx=12, pady=6,
            activebackground=COLOR_ACCENT, activeforeground="white",
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row, text="Capture webcam", command=self._enroll_capture,
            bg=COLOR_PANEL, fg=COLOR_TEXT, relief=tk.FLAT, padx=12, pady=6,
            activebackground=COLOR_ACCENT, activeforeground="white",
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row, text="Enroll", command=self._do_enroll,
            bg=COLOR_ACCENT, fg="white", relief=tk.FLAT, padx=16, pady=6,
            activebackground="#5a7fd4", font=("Helvetica", 11, "bold"),
        ).pack(side=tk.LEFT)

        # Preview
        self._enroll_preview = tk.Label(
            left, text="No image selected", bg=COLOR_PANEL, fg=COLOR_NEUTRAL,
            width=50, height=18, relief=tk.FLAT,
        )
        self._enroll_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self._enroll_status = tk.Label(
            left, text="", fg=COLOR_TEXT, bg=COLOR_BG,
            font=("Helvetica", 11), wraplength=500, justify=tk.LEFT,
        )
        self._enroll_status.pack(anchor=tk.W)

        # Identity list
        tk.Label(
            right, text="Enrolled identities", fg=COLOR_TEXT, bg=COLOR_PANEL,
            font=("Helvetica", 12, "bold"),
        ).pack(anchor=tk.W, padx=12, pady=(12, 8))

        list_frame = tk.Frame(right, bg=COLOR_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._identity_list = tk.Listbox(
            list_frame, bg=COLOR_BG, fg=COLOR_TEXT, relief=tk.FLAT,
            font=("Helvetica", 11), selectbackground=COLOR_ACCENT,
            yscrollcommand=scrollbar.set,
        )
        self._identity_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._identity_list.yview)

        tk.Button(
            right, text="Refresh list", command=self._refresh_identity_list,
            bg=COLOR_BG, fg=COLOR_TEXT, relief=tk.FLAT, padx=8, pady=4,
        ).pack(pady=(0, 12))

    def _build_verify_tab(self, parent: tk.Frame):
        top = tk.Frame(parent, bg=COLOR_BG)
        top.pack(fill=tk.X, pady=(0, 8))

        tk.Button(
            top, text="Upload image", command=self._verify_upload,
            bg=COLOR_PANEL, fg=COLOR_TEXT, relief=tk.FLAT, padx=12, pady=6,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            top, text="Capture webcam", command=self._verify_capture,
            bg=COLOR_PANEL, fg=COLOR_TEXT, relief=tk.FLAT, padx=12, pady=6,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            top, text="Verify", command=self._do_verify,
            bg=COLOR_ACCENT, fg="white", relief=tk.FLAT, padx=16, pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 16))

        self._live_btn = tk.Button(
            top, text="Start live webcam", command=self._toggle_live,
            bg="#16a34a", fg="white", relief=tk.FLAT, padx=12, pady=6,
        )
        self._live_btn.pack(side=tk.LEFT)

        self._fps_label = tk.Label(
            top, text="FPS: —", fg=COLOR_NEUTRAL, bg=COLOR_BG,
            font=("Helvetica", 11),
        )
        self._fps_label.pack(side=tk.RIGHT)

        body = tk.Frame(parent, bg=COLOR_BG)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right = tk.Frame(body, bg=COLOR_PANEL, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._verify_preview = tk.Label(
            left, text="Upload an image or start live webcam",
            bg=COLOR_PANEL, fg=COLOR_NEUTRAL, relief=tk.FLAT,
        )
        self._verify_preview.pack(fill=tk.BOTH, expand=True)

        self._verify_status = tk.Label(
            left, text="", font=("Helvetica", 14, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, wraplength=500, justify=tk.LEFT,
        )
        self._verify_status.pack(anchor=tk.W, pady=(8, 0))

        tk.Label(
            right, text="Confidence scores", fg=COLOR_TEXT, bg=COLOR_PANEL,
            font=("Helvetica", 12, "bold"),
        ).pack(anchor=tk.W, padx=12, pady=(12, 8))

        scores_container = tk.Frame(right, bg=COLOR_PANEL)
        scores_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._scores_canvas = tk.Canvas(
            scores_container, bg=COLOR_PANEL, highlightthickness=0,
        )
        scores_scroll = tk.Scrollbar(scores_container, command=self._scores_canvas.yview)
        self._scores_inner = tk.Frame(self._scores_canvas, bg=COLOR_PANEL)

        self._scores_inner.bind(
            "<Configure>",
            lambda e: self._scores_canvas.configure(
                scrollregion=self._scores_canvas.bbox("all")
            ),
        )
        self._scores_canvas.create_window((0, 0), window=self._scores_inner, anchor=tk.NW)
        self._scores_canvas.configure(yscrollcommand=scores_scroll.set)

        self._scores_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scores_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Enroll actions ────────────────────────────────────────────────────────

    def _enroll_upload(self):
        path = filedialog.askopenfilename(
            title="Select enrollment photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image: {path}")
            return
        self._set_enroll_image(img)

    def _enroll_capture(self):
        if not models_available():
            messagebox.showwarning("Models missing", model_missing_message())
        frame = capture_webcam_frame()
        if frame is None:
            messagebox.showerror("Webcam", "Could not capture from webcam.")
            return
        self._set_enroll_image(frame)

    def _set_enroll_image(self, bgr: np.ndarray):
        self._enroll_image = bgr
        self._enroll_photo = bgr_to_photo(bgr, max_size=(560, 400))
        self._enroll_preview.configure(image=self._enroll_photo, text="")
        self._enroll_status.configure(text="Image ready — click Enroll.", fg=COLOR_NEUTRAL)

    def _do_enroll(self):
        name = self._enroll_name.get().strip()
        if not name:
            self._enroll_status.configure(text="Enter an identity name.", fg=COLOR_FAIL)
            return
        if self._enroll_image is None:
            self._enroll_status.configure(text="Upload or capture a photo first.", fg=COLOR_FAIL)
            return

        result = safe_enroll(name, self._enroll_image)
        if result.get("error") and not result["enrolled"]:
            if "ONNX model" in result["error"]:
                self._enroll_status.configure(text=result["error"], fg="#fbbf24")
                messagebox.showwarning("Models missing", result["error"])
            else:
                self._enroll_status.configure(text=result["error"], fg=COLOR_FAIL)
            return

        if result["enrolled"]:
            self._enroll_status.configure(
                text=f"Enrolled '{result['name']}' — {result['count']} reference photo(s).",
                fg=COLOR_OK,
            )
            self._refresh_identity_list()
        else:
            self._enroll_status.configure(
                text=result.get("error") or "No usable face detected.",
                fg=COLOR_FAIL,
            )

    def _refresh_identity_list(self):
        reload_gallery()
        gallery = get_gallery()
        self._identity_list.delete(0, tk.END)
        identities = gallery.list_identities()
        if not identities:
            self._identity_list.insert(tk.END, "  (none enrolled)")
        else:
            for name in identities:
                count = gallery.count(name)
                self._identity_list.insert(tk.END, f"  {name}  ({count} photo{'s' if count != 1 else ''})")

    # ── Verify actions ────────────────────────────────────────────────────────

    def _verify_upload(self):
        path = filedialog.askopenfilename(
            title="Select image to verify",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image: {path}")
            return
        self._stop_live()
        self._verify_image = img
        self._verify_photo = bgr_to_photo(img, max_size=(560, 400))
        self._verify_preview.configure(image=self._verify_photo, text="")

    def _verify_capture(self):
        frame = capture_webcam_frame()
        if frame is None:
            messagebox.showerror("Webcam", "Could not capture from webcam.")
            return
        self._stop_live()
        self._verify_image = frame
        self._verify_photo = bgr_to_photo(frame, max_size=(560, 400))
        self._verify_preview.configure(image=self._verify_photo, text="")

    def _do_verify(self):
        if self._verify_image is None:
            self._verify_status.configure(text="Select an image first.", fg=COLOR_FAIL)
            return
        result = safe_verify(self._verify_image)
        self._show_verify_result(result)

    def _show_verify_result(self, result: dict):
        if result.get("error"):
            self._verify_status.configure(text=result["error"], fg="#fbbf24")
            self._update_score_bars({})
            if "ONNX model" in result["error"]:
                messagebox.showwarning("Models missing", result["error"])
            return

        if not result["face_found"]:
            self._verify_status.configure(text="No face detected", fg=COLOR_NEUTRAL)
        elif result["verified"]:
            self._verify_status.configure(
                text=f"MATCH: {result['match']}  (confidence {result['confidence']:.2f})",
                fg=COLOR_OK,
            )
        else:
            self._verify_status.configure(
                text=f"No match  (best score {result['confidence']:.2f})",
                fg=COLOR_FAIL,
            )

        self._update_score_bars(result.get("all_scores") or {})

    def _update_score_bars(self, scores: dict):
        for child in self._scores_inner.winfo_children():
            child.destroy()

        if not scores:
            tk.Label(
                self._scores_inner, text="No scores yet",
                fg=COLOR_NEUTRAL, bg=COLOR_PANEL, font=("Helvetica", 10),
            ).pack(anchor=tk.W, pady=4)
            return

        for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            row = tk.Frame(self._scores_inner, bg=COLOR_PANEL)
            row.pack(fill=tk.X, pady=4)

            tk.Label(
                row, text=name, fg=COLOR_TEXT, bg=COLOR_PANEL,
                font=("Helvetica", 10), width=14, anchor=tk.W,
            ).pack(side=tk.LEFT)

            bar_bg = tk.Canvas(row, width=120, height=14, bg=COLOR_BG, highlightthickness=0)
            bar_bg.pack(side=tk.LEFT, padx=(4, 4))
            fill_w = int(max(0, min(1, score)) * 120)
            color = COLOR_OK if score >= config.SIMILARITY_THRESHOLD else COLOR_NEUTRAL
            bar_bg.create_rectangle(0, 0, fill_w, 14, fill=color, outline="")

            tk.Label(
                row, text=f"{score:.2f}", fg=COLOR_TEXT, bg=COLOR_PANEL,
                font=("Helvetica", 10), width=5,
            ).pack(side=tk.LEFT)

    # ── Live webcam ───────────────────────────────────────────────────────────

    def _toggle_live(self):
        if self._live_active:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        self._live_cap = cv2.VideoCapture(0)
        if not self._live_cap.isOpened():
            messagebox.showerror("Webcam", "Could not open webcam.")
            self._live_cap = None
            return

        self._live_active = True
        self._fps_frames = 0
        self._fps_timer = time.time()
        self._display_fps = 0.0
        self._live_btn.configure(text="Stop live webcam", bg=COLOR_FAIL)
        self._verify_status.configure(text="Live mode running…", fg=COLOR_NEUTRAL)
        self._live_tick()

    def _stop_live(self):
        self._live_active = False
        if self._live_job is not None:
            self.after_cancel(self._live_job)
            self._live_job = None
        if self._live_cap is not None:
            self._live_cap.release()
            self._live_cap = None
        self._live_btn.configure(text="Start live webcam", bg="#16a34a")
        self._fps_label.configure(text="FPS: —")

    def _live_tick(self):
        if not self._live_active or self._live_cap is None:
            return

        ret, frame = self._live_cap.read()
        if not ret:
            self._stop_live()
            messagebox.showerror("Webcam", "Lost webcam feed.")
            return

        result = safe_verify(frame)
        if result.get("error") and "ONNX model" in result["error"]:
            self._stop_live()
            self._verify_status.configure(text=result["error"], fg="#fbbf24")
            messagebox.showwarning("Models missing", result["error"])
            return

        display = _draw_overlay(frame.copy(), result)
        self._live_photo = bgr_to_photo(display, max_size=(640, 480))
        self._verify_preview.configure(image=self._live_photo, text="")
        self._show_verify_result(result)

        self._fps_frames += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._display_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_timer = time.time()
        self._fps_label.configure(text=f"FPS: {self._display_fps:.1f}")

        self._live_job = self.after(30, self._live_tick)

    def destroy(self):
        self._stop_live()
        super().destroy()


def main():
    app = FaceIdApp()
    app.mainloop()


if __name__ == "__main__":
    main()
