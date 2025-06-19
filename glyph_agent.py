# glyph_agent.py

from typing import List, Optional
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import json

# --- Finalized DTO: FeltDTO_v1.1 ---
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
        consent_handshake_id_ref: Optional[str] = None,
        mesh_signature: Optional[str] = None,
        source_weave_id_ref: Optional[str] = None,
        source_template_id_ref: Optional[str] = None
    ):
        self.id = id or uuid4().hex
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.author = author
        self.content_truth = content_truth
        self.vector_embedding = vector_embedding
        self.ache_scalar = ache_scalar
        self.qualia_tags = qualia_tags
        self.archetypes = archetypes
        self.consent_handshake_id_ref = consent_handshake_id_ref
        self.mesh_signature = mesh_signature
        self.source_weave_id_ref = source_weave_id_ref
        self.source_template_id_ref = source_template_id_ref

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
            "consent_handshake_id_ref": self.consent_handshake_id_ref,
            "mesh_signature": self.mesh_signature,
            "source_weave_id_ref": self.source_weave_id_ref,
            "source_template_id_ref": self.source_template_id_ref
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
            consent_handshake_id_ref=data.get("consent_handshake_id_ref"),
            mesh_signature=data.get("mesh_signature"),
            source_weave_id_ref=data.get("source_weave_id_ref"),
            source_template_id_ref=data.get("source_template_id_ref")
        )

# --- Core Memory Engine: GlyphAgent ---
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

# --- Example Embedding Function (replace with Gemini/MiniLM/Faiss) ---
def fake_embed(text: str):
    arr = np.zeros(128)
    for i, c in enumerate(text.encode()[:128]):
        arr[i] = float(c) / 255
    return arr

# --- Example Usage ---
if __name__ == "__main__":
    agent = GlyphAgent(fake_embed)
    agent.store(FeltDTO(
        content_truth="The ache of your gaze never left my skin.",
        qualia_tags=["ache", "gaze"],
        archetypes=["longing"],
        vector_embedding=fake_embed("The ache of your gaze never left my skin."),
        ache_scalar=1.0
    ))
    agent.store(FeltDTO(
        content_truth="A gasp between words, and presence deepened.",
        qualia_tags=["gasp", "presence"],
        archetypes=["awakening"],
        vector_embedding=fake_embed("A gasp between words, and presence deepened."),
        ache_scalar=0.92
    ))
    prompt_context = agent.hydrate("Describe how it feels to long for someone's presence.")
    print("Hydrated Prompt:\n", "\n".join(prompt_context))
