"""
verify.py — CLI tool to verify a face against the enrolled gallery.

Usage:
    # Verify a single image
    python verify.py photo.jpg

    # Run live webcam verification loop (press Q to quit)
    python verify.py --live

    # Adjust threshold for this run
    python verify.py photo.jpg --threshold 0.5
"""

import argparse
import sys
import time
import cv2
import numpy as np

from face_id.pipeline import verify, verify_from_array
from face_id import config


def verify_image(path: str, threshold: float):
    result = verify(path, threshold=threshold)
    _print_result(result, source=path)


def run_live(threshold: float):
    """
    Open the default webcam and run continuous face verification.
    Draws the result on-screen. Press Q to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print("Running live verification. Press Q to quit.")
    fps_timer = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = verify_from_array(frame, threshold=threshold)
        frame = _draw_overlay(frame, result)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()
        else:
            fps = 0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("face_id — live verify", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    h, w = frame.shape[:2]

    if not result["face_found"]:
        label = "No face detected"
        color = (100, 100, 100)
    elif result["verified"]:
        label = f"MATCH: {result['match']}  ({result['confidence']:.2f})"
        color = (0, 220, 0)    # green
    else:
        label = f"NO MATCH  ({result['confidence']:.2f})"
        color = (0, 60, 220)   # red

    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def _print_result(result: dict, source: str):
    print(f"\nSource:     {source}")
    print(f"Face found: {result['face_found']}")
    if result.get("error"):
        print(f"Error:      {result['error']}")
        return
    print(f"Verified:   {result['verified']}")
    print(f"Best match: {result['match']}")
    print(f"Confidence: {result['confidence']}")
    if result.get("all_scores"):
        print("All scores:")
        for name, score in sorted(result["all_scores"].items(),
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {name:<20} {score:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify a face against the enrolled gallery."
    )
    parser.add_argument("image", nargs="?", help="Image file path to verify")
    parser.add_argument("--live", action="store_true", help="Run webcam live mode")
    parser.add_argument(
        "--threshold", type=float, default=config.SIMILARITY_THRESHOLD,
        help=f"Similarity threshold (default: {config.SIMILARITY_THRESHOLD})"
    )
    args = parser.parse_args()

    if args.live:
        run_live(args.threshold)
    elif args.image:
        verify_image(args.image, args.threshold)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
