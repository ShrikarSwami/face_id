"""
tests/test_verify.py — unit tests for the verify pipeline.

These tests mock the model calls so they run without ONNX weights.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_unit_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


class TestPipeline:

    @patch("face_id.pipeline.detect_faces")
    @patch("face_id.pipeline.align_face")
    @patch("face_id.pipeline.compute_embedding")
    @patch("face_id.pipeline.get_gallery")
    def test_verify_match(self, mock_gallery, mock_embed, mock_align, mock_detect):
        """verify() returns verified=True when embedding matches enrolled identity."""
        from face_id.pipeline import verify_from_array
        from face_id.gallery_manager import Gallery

        alice_emb = _make_unit_embedding(seed=1)

        mock_detect.return_value = [{"bbox": [0, 0, 100, 100], "landmarks": np.zeros((5, 2)), "confidence": 0.9}]
        mock_align.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_embed.return_value = alice_emb

        gallery = MagicMock()
        gallery.get_prototypes.return_value = {"alice": alice_emb}
        mock_gallery.return_value = gallery

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verify_from_array(frame)

        assert result["face_found"] is True
        assert result["verified"] is True
        assert result["match"] == "alice"
        assert result["confidence"] > 0.99

    @patch("face_id.pipeline.detect_faces")
    @patch("face_id.pipeline.get_gallery")
    def test_verify_no_face(self, mock_gallery, mock_detect):
        """verify() returns face_found=False when detector finds nothing."""
        from face_id.pipeline import verify_from_array

        mock_detect.return_value = []
        gallery = MagicMock()
        gallery.get_prototypes.return_value = {"alice": _make_unit_embedding(1)}
        mock_gallery.return_value = gallery

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verify_from_array(frame)

        assert result["face_found"] is False
        assert result["verified"] is False

    @patch("face_id.pipeline.detect_faces")
    @patch("face_id.pipeline.align_face")
    @patch("face_id.pipeline.compute_embedding")
    @patch("face_id.pipeline.get_gallery")
    def test_verify_wrong_person(self, mock_gallery, mock_embed, mock_align, mock_detect):
        """verify() returns verified=False when embeddings are orthogonal."""
        from face_id.pipeline import verify_from_array

        alice_emb = np.zeros(512, dtype=np.float32)
        alice_emb[0] = 1.0
        query_emb = np.zeros(512, dtype=np.float32)
        query_emb[1] = 1.0  # orthogonal — different person

        mock_detect.return_value = [{"bbox": [0, 0, 100, 100], "landmarks": np.zeros((5, 2)), "confidence": 0.9}]
        mock_align.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_embed.return_value = query_emb

        gallery = MagicMock()
        gallery.get_prototypes.return_value = {"alice": alice_emb}
        mock_gallery.return_value = gallery

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verify_from_array(frame, threshold=0.45)

        assert result["verified"] is False
        assert result["match"] is None

    @patch("face_id.pipeline.detect_faces")
    @patch("face_id.pipeline.align_face")
    @patch("face_id.pipeline.compute_embedding")
    @patch("face_id.pipeline.get_gallery")
    def test_threshold_override(self, mock_gallery, mock_embed, mock_align, mock_detect):
        """verify() respects threshold override argument."""
        from face_id.pipeline import verify_from_array

        # Embeddings with similarity ~0.5
        a = np.zeros(512, dtype=np.float32)
        a[0] = 0.7071
        a[1] = 0.7071
        b = np.zeros(512, dtype=np.float32)
        b[0] = 1.0

        mock_detect.return_value = [{"bbox": [0, 0, 100, 100], "landmarks": np.zeros((5, 2)), "confidence": 0.9}]
        mock_align.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_embed.return_value = a

        gallery = MagicMock()
        gallery.get_prototypes.return_value = {"alice": b}
        mock_gallery.return_value = gallery

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should match with loose threshold
        result_loose = verify_from_array(frame, threshold=0.3)
        assert result_loose["verified"] is True

        # Should not match with strict threshold
        result_strict = verify_from_array(frame, threshold=0.9)
        assert result_strict["verified"] is False
