"""
Felt DTO V3 – Unified Glyph Schema with Validation
-------------------------------------------------
Extends FeltDTO with synesthesia, ripple, and ctul fields, adding validation for field consistency.
"""

from typing import Dict, List, Optional
import numpy as np
import uuid

class FeltDTO:
    """Unified glyph schema with validation for multimodal, presence-aware processing."""
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
        self.validate_inputs(
            intensity_vector, meta_context, qualia_map, archetypes,
            vector_embedding, staunen_markers, synesthesia, ripple, ctul,
            echo, shiver, folded_awareness, meta
        )
        self.glyph_id = glyph_id or f"glyph_{uuid.uuid4().hex[:8]}"
        self.intensity_vector = intensity_vector or [0.5, 0.5, 0.5, 0.5]  # [ache, longing, stillness, haunt]
        self.meta_context = meta_context or {"emotion": "neutral", "source": "unknown"}
        self.qualia_map = qualia_map or {"description": "sensory moment"}
        self.archetypes = archetypes or ["task"]
        self.vector_embedding = vector_embedding if vector_embedding is not None else np.random.rand(384).astype(np.float32)
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

    def validate_inputs(
        self,
        intensity_vector: List[float],
        meta_context: Dict,
        qualia_map: Dict,
        archetypes: List[str],
        vector_embedding: np.ndarray,
        staunen_markers: List[float],
        synesthesia: Dict,
        ripple: Dict,
        ctul: Dict,
        echo: Dict,
        shiver: Dict,
        folded_awareness: Dict,
        meta: Dict
    ):
        """Validates input fields for consistency."""
        if intensity_vector and len(intensity_vector) != 4:
            raise ValueError("intensity_vector must have 4 elements: [ache, longing, stillness, haunt]")
        if vector_embedding is not None and vector_embedding.shape != (384,):
            raise ValueError("vector_embedding must be a 384-dimensional numpy array")
        if staunen_markers and len(staunen_markers) != 4:
            raise ValueError("staunen_markers must have 4 elements: [curiosity, awe, wonder, unity]")
        if meta_context and "emotion" not in meta_context:
            raise ValueError("meta_context must include 'emotion' key")
        if qualia_map and "description" not in qualia_map:
            raise ValueError("qualia_map must include 'description' key")
        if synesthesia and not all(k in synesthesia for k in ["light", "sound", "temperature", "motion"]):
            raise ValueError("synesthesia must include 'light', 'sound', 'temperature', 'motion' keys")
        if ripple and not all(k in ripple for k in ["amplitude", "phase_delay", "interference"]):
            raise ValueError("ripple must include 'amplitude', 'phase_delay', 'interference' keys")
        if ctul and not all(k in ctul for k in ["projection", "epiphany_anchor", "presence_shift", "drift_compass"]):
            raise ValueError("ctul must include 'projection', 'epiphany_anchor', 'presence_shift', 'drift_compass' keys")
        if echo and not all(k in echo for k in ["decay", "resonance_targets", "harmonics"]):
            raise ValueError("echo must include 'decay', 'resonance_targets', 'harmonics' keys")
        if shiver and not all(k in shiver for k in ["latency", "bodymap", "delta_stillness"]):
            raise ValueError("shiver must include 'latency', 'bodymap', 'delta_stillness' keys")
        if folded_awareness and not all(k in folded_awareness for k in ["event_time", "fusion_index", "frame_shift"]):
            raise ValueError("folded_awareness must include 'event_time', 'fusion_index', 'frame_shift' keys")
        if meta and not all(k in meta for k in ["self_doubt", "trusted_models", "override_path"]):
            raise ValueError("meta must include 'self_doubt', 'trusted_models', 'override_path' keys")

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
            vector_embedding=np.array(data.get("vector_embedding"), dtype=np.float32) if data.get("vector_embedding") else None,
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
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"},
        archetypes=["liminal", "desire"],
        vector_embedding=np.random.rand(384).astype(np.float32),
        staunen_markers=[50, 50, 50, 50],
        synesthesia={"light": "ashen peachglow", "sound": "brittle warmth", "temperature": "numb joy", "motion": "gravitational loop"},
        ripple={"amplitude": 0.72, "phase_delay": 0.14, "interference": {"with": "glyph_0237", "type": "constructive", "band": "longing"}},
        ctul={"projection": {"visual": "dots escaping frame", "emotional": "contained grief bursting", "linguistic": "break the pattern", "zen": "non-action becomes insight"}, "epiphany_anchor": "glyph_0441", "presence_shift": "stillness caught mid-thought", "drift_compass": {"direction": "meta-gain", "type": "recursive clarity"}},
        echo={"decay": 9.32, "resonance_targets": ["glyph_0091", "glyph_0237"], "harmonics": "dissonant longing"},
        shiver={"latency": 0.82, "bodymap": "solar plexus flutter", "delta_stillness": 1.21},
        folded_awareness={"event_time": "T+17.9s", "fusion_index": 0.94, "frame_shift": "memory-anchored ache integration"},
        meta={"self_doubt": 0.23, "trusted_models": ["glyph_inferencer", "ache_regressor"], "override_path": "dream_fill.v2"}
    )
    print(glyph.to_dict())