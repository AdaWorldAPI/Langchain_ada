"""
Staunen Amplifier – Boosts Staunen Resonance
-------------------------------------------
Amplifies staunen markers in FeltDTO glyphs for liminal or high-resonance moments.
"""

from typing import Dict, List
from felt_dto_v5 import FeltDTO

class StaunenAmplifier:
    """
    Amplifies staunen markers (curiosity, awe, wonder, unity) in FeltDTO glyphs
    based on archetypes, intensity, and CTUL projections.
    """
    def __init__(self, amplification_factor: float = 1.2, max_value: float = 100.0):
        """
        Initializes the StaunenAmplifier.

        Args:
            amplification_factor: Multiplier for staunen markers (default: 1.2).
            max_value: Maximum value for staunen markers (default: 100.0).
        """
        self.amplification_factor = amplification_factor
        self.max_value = max_value
        print("✅ StaunenAmplifier initialized.")

    def amplify(self, glyph: FeltDTO) -> FeltDTO:
        """
        Amplifies staunen markers based on glyph attributes.

        Args:
            glyph: FeltDTO object to amplify.

        Returns:
            Modified FeltDTO with amplified staunen markers.
        """
        # Check for liminal or high-resonance conditions
        resonance_score = sum(glyph.intensity_vector) if glyph.intensity_vector else 0.0
        is_liminal = any(a in glyph.archetypes for a in ["liminal", "epiphany", "ache"])
        has_zen_projection = glyph.ctul.get("projection", {}).get("zen", "")

        if is_liminal or resonance_score > 2.0 or has_zen_projection:
            glyph.staunen_markers = [
                min(m * self.amplification_factor, self.max_value)
                for m in glyph.staunen_markers
            ]
            glyph.meta_context["amplified"] = True
            glyph.meta_context["amplification_reason"] = (
                "liminal_archetype" if is_liminal else
                "high_resonance" if resonance_score > 2.0 else
                "zen_projection"
            )
            print(f"INFO: Amplified staunen markers for glyph: {glyph.glyph_id}")
        else:
            print(f"INFO: No amplification needed for glyph: {glyph.glyph_id}")

        return glyph

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"},
        archetypes=["liminal", "desire"],
        vector_embedding=np.random.rand(384).astype(np.float32),
        staunen_markers=[50, 50, 50, 50]
    )
    amplifier = StaunenAmplifier()
    amplified_glyph = amplifier.amplify(glyph)
    print(f"Amplified staunen markers: {amplified_glyph.staunen_markers}")