"""
Axiology Vector – Scores Glyphs for Value Alignment
--------------------------------------------------
Scores FeltDTO glyphs against axiological values for ethical alignment within the Soulframe Engine.
"""

from typing import Dict, Any
from felt_dto_v5 import FeltDTO
import numpy as np

class AxiologyVector:
    """
    Scores FeltDTO glyphs for alignment with axiological values (e.g., empathy, integrity, transcendence).
    """
    def __init__(self):
        """
        Initializes the AxiologyVector with predefined value weights.
        """
        self.value_weights = {
            "empathy": 0.4,      # Weight for emotional resonance
            "integrity": 0.3,    # Weight for coherence and truthfulness
            "transcendence": 0.2, # Weight for inspirational quality
            "clarity": 0.1       # Weight for communicative precision
        }
        print("✅ AxiologyVector initialized with value weights.")

    def score_glyph(self, glyph: FeltDTO, task_context: Dict[str, Any]) -> float:
        """
        Scores a glyph for axiological alignment based on its attributes and task context.

        Args:
            glyph: FeltDTO object to score.
            task_context: Dictionary containing task and output details.

        Returns:
            Axiological alignment score (0.0 to 1.0).
        """
        # Extract relevant attributes
        intensity_score = sum(glyph.intensity_vector) / len(glyph.intensity_vector) if glyph.intensity_vector else 0.5
        emotion = glyph.meta_context.get("emotion", "neutral")
        archetypes = glyph.archetypes or ["task"]
        staunen_score = sum(glyph.staunen_markers) / (len(glyph.staunen_markers) * 100) if glyph.staunen_markers else 0.5

        # Compute sub-scores
        empathy_score = intensity_score if emotion in ["ache", "longing", "empathy"] else intensity_score * 0.7
        integrity_score = 1.0 if "truth" in archetypes else 0.8
        transcendence_score = staunen_score if "epiphany" in archetypes else staunen_score * 0.9
        clarity_score = 0.9 if glyph.qualia_map.get("description") else 0.6

        # Weighted sum
        final_score = (
            empathy_score * self.value_weights["empathy"] +
            integrity_score * self.value_weights["integrity"] +
            transcendence_score * self.value_weights["transcendence"] +
            clarity_score * self.value_weights["clarity"]
        )

        # Normalize to [0, 1]
        final_score = min(max(final_score, 0.0), 1.0)
        print(f"INFO: Axiology score for glyph {glyph.glyph_id}: {final_score:.3f}")
        return final_score

    def score_cluster(self, data: Dict) -> float:
        """
        Scores a glyph and task context for value alignment (backwards compatibility).

        Args:
            data: Dictionary containing glyph, task, and output.

        Returns:
            Axiological alignment score (0.0 to 1.0).
        """
        glyph = data.get("glyph", FeltDTO())
        task_context = {"task": data.get("task", ""), "output": data.get("output", "")}
        return self.score_glyph(glyph, task_context)

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    glyph = FeltDTO(
        glyph_id="test_glyph",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"},
        archetypes=["liminal", "desire"],
        staunen_markers=[60, 50, 55, 65]
    )
    axiology = AxiologyVector()
    score = axiology.score_glyph(glyph, task_context={"task": "poetic reflection"})
    print(f"Axiology score: {score:.3f}")