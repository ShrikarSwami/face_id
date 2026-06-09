"""
tests/test_enroll.py — unit tests for enrollment pipeline.

Run with: pytest face_id/tests/
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestGallery:
    """Tests for the Gallery class without requiring model files."""

    def test_add_and_count(self, tmp_path):
        from face_id.gallery_manager import Gallery
        g = Gallery(gallery_dir=tmp_path)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        g.add("alice", emb)
        assert g.count("alice") == 1

        g.add("alice", emb)
        assert g.count("alice") == 2

    def test_save_and_load(self, tmp_path):
        from face_id.gallery_manager import Gallery
        g = Gallery(gallery_dir=tmp_path)

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        g.add("bob", emb)
        g.save()

        g2 = Gallery(gallery_dir=tmp_path)
        g2.load()
        assert "bob" in g2.list_identities()
        assert g2.count("bob") == 1

    def test_prototype_is_unit_norm(self, tmp_path):
        from face_id.gallery_manager import Gallery
        g = Gallery(gallery_dir=tmp_path)

        for _ in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            g.add("carol", emb)

        protos = g.get_prototypes()
        norm = np.linalg.norm(protos["carol"])
        assert abs(norm - 1.0) < 1e-5

    def test_remove(self, tmp_path):
        from face_id.gallery_manager import Gallery
        g = Gallery(gallery_dir=tmp_path)

        emb = np.random.randn(512).astype(np.float32)
        g.add("dave", emb)
        g.save()

        removed = g.remove("dave")
        assert removed is True
        assert "dave" not in g.list_identities()
        assert not (tmp_path / "dave.npy").exists()


class TestSimilarity:
    """Tests for cosine similarity and matching logic."""

    def test_identical_vectors(self):
        from face_id.utils.similarity import cosine_similarity
        v = np.random.randn(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        from face_id.utils.similarity import cosine_similarity
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_best_match_returns_correct_identity(self):
        from face_id.utils.similarity import best_match
        alice = np.zeros(512, dtype=np.float32)
        alice[0] = 1.0
        bob = np.zeros(512, dtype=np.float32)
        bob[1] = 1.0

        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0  # identical to alice

        gallery = {"alice": alice, "bob": bob}
        result = best_match(query, gallery, threshold=0.9)

        assert result["verified"] is True
        assert result["match"] == "alice"
        assert result["confidence"] > 0.99

    def test_no_match_below_threshold(self):
        from face_id.utils.similarity import best_match
        alice = np.zeros(512, dtype=np.float32)
        alice[0] = 1.0

        query = np.zeros(512, dtype=np.float32)
        query[1] = 1.0  # orthogonal to alice

        gallery = {"alice": alice}
        result = best_match(query, gallery, threshold=0.5)

        assert result["verified"] is False
        assert result["match"] is None

    def test_empty_gallery(self):
        from face_id.utils.similarity import best_match
        query = np.random.randn(512).astype(np.float32)
        result = best_match(query, {}, threshold=0.45)
        assert result["verified"] is False
        assert result["match"] is None
