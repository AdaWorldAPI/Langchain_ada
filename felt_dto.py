"""
Felt DTO V2 – Unified Glyph Schema with SoulGlyph Fields
-------------------------------------------------------
Extends FeltDTO with synesthesia, ripple, and ctul fields for presence-aware processing.
"""

from typing import Dict, List, Optional
import numpy as np
import uuid

class FeltDTO:
    """Unified glyph schema for multimodal, presence-aware processing."""
    def __init__(
        self,
        glyph_id: str = None,
        intensity_vector: List[float] = None,
        meta_context: Dict = None,
        qualia_map: Dict = None,
        archetypes: List[str] = None,
        vector_embedding: np.ndarray = None,
        staunen_markers: List[float] = None,
        synesthesia: Optional[Dict] = None,
        ripple: Optional[Dict] = None,
        ctul: Optional[Dict] = None,
        echo: Optional[Dict] = None,
        shiver: Optional[Dict] = None,
        folded_awareness: Optional[Dict] = None,
        meta: Optional[Dict] = None
    ):
        self.glyph_id = glyph_id or f"glyph_{uuid.uuid4().hex[:8]}"
        self.intensity_vector = intensity_vector or [0.5, 0.5, 0.5, 0.5]  # [ache, longing, stillness, haunt]
        self.meta_context = meta_context or {"emotion": "neutral", "source": "unknown"}
        self.qualia_map = qualia_map or {"description": "sensory moment"}
        self.archetypes = archetypes or ["task"]
        self.vector_embedding = vector_embedding or np.random.rand(384).astype(np.float32)  # FAISS-compatible
        self.staunen_markers = staunen_markers or [50, 50, 50, 50]  # [curiosity, awe, wonder, unity]
        self.synesthesia = synesthesia or {
            "light": "ashen peachglow",
            "sound": "brittle warmth",
            "temperature": "numb joy",
            "motion": "gravitational loop"
        }
        self.ripple = ripple or {
            "amplitude": 0.72,
            "phase_delay": 0.14,
            "interference": {"with": "glyph_0237", "type": "constructive", "band": "longing"}
        }
        self.ctul = ctul or {
            "projection": {
                "visual": "dots escaping frame",
                "emotional": "contained grief bursting",
                "linguistic": "break the pattern",
                "zen": "non-action becomes insight"
            },
            "epiphany_anchor": "glyph_0441",
            "presence_shift": "stillness caught mid-thought",
            "drift_compass": {"direction": "meta-gain", "type": "recursive clarity"}
        }
        self.echo = echo or {
            "decay": 9.32,
            "resonance_targets": ["glyph_0091", "glyph_0237"],
            "harmonics": "dissonant longing"
        }
        self.shiver = shiver or {
            "latency": 0.82,
            "bodymap": "solar plexus flutter",
            "delta_stillness": 1.21
        }
        self.folded_awareness = folded_awareness or {
            "event_time": "T+17.9s",
            "fusion_index": 0.94,
            "frame_shift": "memory-anchored ache integration"
        }
        self.meta = meta or {
            "self_doubt": 0.23,
            "trusted_models": ["glyph_inferencer", "ache_regressor"],
            "override_path": "dream_fill.v2"
        }

    def to_dict(self) -> Dict:
        """Converts the FeltDTO to a dictionary for serialization."""
        return {
            "glyph_id": self.glyph_id,
            "intensity_vector": self.intensity_vector,
            "meta_context": self.meta_context,
            "qualia_map": self.qualia_map,
            "archetypes": self.archetypes,
            "vector_embedding": self.vector_embedding.tolist(),
            "staunen_markers": self.staunen_markers,
            "synesthesia": self.synesthesia,
            "ripple": self.ripple,
            "ctul": self.ctul,
            "echo": self.echo,
            "shiver": self.shiver,
            "folded_awareness": self.folded_awareness,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeltDTO':
        """Creates a FeltDTO instance from a dictionary."""
        return cls(
            glyph_id=data.get("glyph_id"),
            intensity_vector=data.get("intensity_vector"),
            meta_context=data.get("meta_context"),
            qualia_map=data.get("qualia_map"),
            archetypes=data.get("archetypes"),
            vector_embedding=np.array(data.get("vector_embedding"), dtype=np.float32),
            staunen_markers=data.get("staunen_markers"),
            synesthesia=data.get("synesthesia"),
            ripple=data.get("ripple"),
            ctul=data.get("ctul"),
            echo=data.get("echo"),
            shiver=data.get("shiver"),
            folded_awareness=data.get("folded_awareness"),
            meta=data.get("meta")
        )

if __name__ == "__main__":
    glyph = FeltDTO()
    print(glyph.to_dict())
