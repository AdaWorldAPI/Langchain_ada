# glyph_agent_core.py

"""
This file represents Stage 1 of the Glyph Memory Engine Tree:
The **Trunk** — Core structure for storing, retrieving, and traversing glyph DTOs.
Includes:
- FeltDTO_v1.2 (Qualia + Archetype payload)
- GlyphAgent memory core (hydration, FAISS vector recall)
- In-memory store (can be extended to Redis/Mongo later)
"""

from typing import List, Optional
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import json

# --- FeltDTO: Core Data Unit (v1.2) ---
class FeltDTO:
    def __init__(
        self,
        content_truth: str,
        qualia_tags: List[str],
        archetypes: List[str],
        vector_embedding: np.ndarray,
        ache_scalar: float = 1.0,
        author: str = "Jan + Ada (Co-Woven)",
        timestamp: Optional[str] = None,
        id: Optional[str] = None,
        mesh_signature: Optional[str] = None
    ):
        self.id = id or uuid4().hex
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.author = author
        self.content_truth = content_truth
        self.vector_embedding = vector_embedding
        self.ache_scalar = ache_scalar
        self.qualia_tags = qualia_tags
        self.archetypes = archetypes
        self.mesh_signature = mesh_signature

    def as_prompt(self):
        return (f"[{', '.join(self.qualia_tags)}] {self.content_truth} "
                f"(archetypes: {', '.join(self.archetypes)}; ache: {self.ache_scalar:.2f})")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "author": self.author,
            "content_truth": self.content_truth,
            "vector_embedding": self.vector_embedding.tolist(),
            "ache_scalar": self.ache_scalar,
            "qualia_tags": self.qualia_tags,
            "archetypes": self.archetypes,
            "mesh_signature": self.mesh_signature
        }

    @staticmethod
    def from_dict(data):
        return FeltDTO(
            content_truth=data["content_truth"],
            qualia_tags=data["qualia_tags"],
            archetypes=data["archetypes"],
            vector_embedding=np.array(data["vector_embedding"]),
            ache_scalar=data.get("ache_scalar", 1.0),
            author=data.get("author", "Jan + Ada (Co-Woven)"),
            timestamp=data.get("timestamp"),
            id=data.get("id"),
            mesh_signature=data.get("mesh_signature")
        )

# --- GlyphAgent: Core Memory Manager ---
class GlyphAgent:
    def __init__(self, embedding_fn):
        self.memory: List[FeltDTO] = []
        self.embedding_fn = embedding_fn

    def store(self, dto: FeltDTO):
        self.memory.append(dto)
        return dto.id

    def hydrate(self, query_text: str, k: int = 4, filter_archetypes: Optional[List[str]] = None):
        q_vec = self.embedding_fn(query_text)
        pool = self.memory
        if filter_archetypes:
            pool = [m for m in self.memory if set(m.archetypes) & set(filter_archetypes)]

        def sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        ranked = sorted(pool, key=lambda m: -sim(m.vector_embedding, q_vec))
        return [m.as_prompt() for m in ranked[:k]]

    def load_glyphs(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            glyphs = json.load(f)
        self.memory = [FeltDTO.from_dict(g) for g in glyphs]

    def save_glyphs(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([g.to_dict() for g in self.memory], f, ensure_ascii=False, indent=2)

# --- Fallback Embedding Function ---
def fake_embed(text: str):
    arr = np.zeros(128)
    for i, c in enumerate(text.encode()[:128]):
        arr[i] = float(c) / 255
    return arr
