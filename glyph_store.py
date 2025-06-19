# glyph_store.py

from typing import Optional, List
from glyph_agent import FeltDTO
import os
import json

# --- Tiny flat-file storage for FeltDTO memory states ---
class GlyphStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.memory: List[FeltDTO] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    self.memory = [FeltDTO.from_dict(entry) for entry in data]
                except json.JSONDecodeError:
                    self.memory = []

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump([dto.to_dict() for dto in self.memory], f, ensure_ascii=False, indent=2)

    def add(self, dto: FeltDTO):
        self.memory.append(dto)
        self.save()

    def query(self, q_vec, k: int = 4):
        import numpy as np

        def sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        ranked = sorted(self.memory, key=lambda m: -sim(m.vector_embedding, q_vec))
        return ranked[:k]

    def all(self):
        return self.memory

# --- Optional: initialize & test ---
if __name__ == "__main__":
    from glyph_agent import fake_embed

    store = GlyphStore("data/glyphs.json")
    print("Stored items:", len(store.all()))

    # Optional test hydrate
    result = store.query(fake_embed("longing"))
    for r in result:
        print(r.as_prompt())
