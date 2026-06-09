"""
gallery.py — manages the enrolled identity gallery on disk.

Each identity is stored as a .npy file containing either:
  - a single 512-dim embedding vector, or
  - a stacked (N, 512) array of multiple reference embeddings

The gallery is loaded into memory as a dict of prototype vectors
for fast comparison at inference time.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from face_id import config
from face_id.utils.similarity import make_prototype


class Gallery:
    """
    Manages the on-disk enrolled identity store.

    Usage:
        gallery = Gallery()
        gallery.add("alice", embedding)          # add one embedding
        gallery.add("alice", another_embedding)  # adds to alice's set
        gallery.save()

        gallery.load()
        prototypes = gallery.get_prototypes()    # {name: mean_embedding}
    """

    def __init__(self, gallery_dir: Path = None):
        self._dir = gallery_dir or config.GALLERY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        # In-memory store: {name: [emb1, emb2, ...]}
        self._embeddings: Dict[str, List[np.ndarray]] = {}

    # ── Writing ───────────────────────────────────────────────────────────────

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Add one embedding to an identity's reference set."""
        if name not in self._embeddings:
            self._embeddings[name] = []
        self._embeddings[name].append(embedding.astype(np.float32))

    def save(self, name: str = None) -> None:
        """
        Persist embeddings to disk.

        Args:
            name: If given, save only this identity. Otherwise save all.
        """
        targets = [name] if name else list(self._embeddings.keys())
        for n in targets:
            if n in self._embeddings:
                stacked = np.stack(self._embeddings[n], axis=0)
                np.save(self._dir / f"{n}.npy", stacked)

    # ── Reading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all identity embeddings from disk into memory."""
        self._embeddings = {}
        for path in sorted(self._dir.glob("*.npy")):
            name = path.stem
            data = np.load(path)
            # Stored as (N, 512) or (512,) for legacy single embeddings
            if data.ndim == 1:
                data = data[np.newaxis, :]
            self._embeddings[name] = [data[i] for i in range(len(data))]

    def get_prototypes(self) -> Dict[str, np.ndarray]:
        """
        Return {name: prototype_embedding} for all enrolled identities.
        Prototype = L2-normalised mean of all reference embeddings.
        """
        return {
            name: make_prototype(embs)
            for name, embs in self._embeddings.items()
            if embs
        }

    # ── Management ────────────────────────────────────────────────────────────

    def list_identities(self) -> List[str]:
        """Return names of all enrolled identities."""
        return sorted(self._embeddings.keys())

    def remove(self, name: str) -> bool:
        """Remove an identity from memory and disk. Returns True if found."""
        removed = False
        if name in self._embeddings:
            del self._embeddings[name]
            removed = True
        path = self._dir / f"{name}.npy"
        if path.exists():
            path.unlink()
            removed = True
        return removed

    def count(self, name: str) -> int:
        """Return the number of reference embeddings for an identity."""
        return len(self._embeddings.get(name, []))

    def __len__(self) -> int:
        return len(self._embeddings)

    def __repr__(self) -> str:
        summary = {n: len(e) for n, e in self._embeddings.items()}
        return f"Gallery({summary})"


# Module-level singleton loaded from disk once
_gallery: Optional[Gallery] = None


def get_gallery() -> Gallery:
    global _gallery
    if _gallery is None:
        _gallery = Gallery()
        _gallery.load()
    return _gallery


def reload_gallery() -> Gallery:
    """Force a reload from disk — call after external changes."""
    global _gallery
    _gallery = Gallery()
    _gallery.load()
    return _gallery
