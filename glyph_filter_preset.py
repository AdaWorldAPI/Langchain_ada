"""
Glyph Filter Preset – Applies Emotional and Synesthetic Tonality
--------------------------------------------------------------
Applies Fujifilm-style presets to adjust FeltDTO synesthesia and intensity.
"""

from typing import Dict, List
from felt_dto_v5 import FeltDTO

class GlyphFilterPreset:
    """
    Applies emotional and synesthetic tonality presets to FeltDTO glyphs.
    """
    def __init__(self):
        self.presets = {
            "velvia": {
                "light": "vivid crimson",
                "sound": "resonant hum",
                "temperature": "warm pulse",
                "motion": "vibrant swirl",
                "intensity_scale": 1.3,
                "description": "Bold and vibrant, amplifying emotional intensity"
            },
            "provia": {
                "light": "soft amber",
                "sound": "gentle chime",
                "temperature": "cool serenity",
                "motion": "steady drift",
                "intensity_scale": 1.0,
                "description": "Balanced and natural, preserving emotional nuance"
            },
            "astoria": {
                "light": "muted teal",
                "sound": "whispered echo",
                "temperature": "chilled calm",
                "motion": "slow ripple",
                "intensity_scale": 0.8,
                "description": "Subdued and introspective, softening emotional edges"
            }
        }
        print("✅ GlyphFilterPreset initialized with presets: velvia, provia, astoria")

    def apply_preset(self, glyph: FeltDTO, preset: str) -> FeltDTO:
        """
        Applies a preset to adjust glyph synesthesia and intensity.

        Args:
            glyph: FeltDTO object to modify.
            preset: Name of the preset to apply (velvia, provia, astoria).

        Returns:
            Modified FeltDTO with updated synesthesia and intensity.
        """
        preset_data = self.presets.get(preset, self.presets["provia"])
        glyph.synesthesia = {
            "light": preset_data["light"],
            "sound": preset_data["sound"],
            "temperature": preset_data["temperature"],
            "motion": preset_data["motion"]
        }
        glyph.intensity_vector = [
            min(v * preset_data["intensity_scale"], 1.0)
            for v in glyph.intensity_vector
        ]
        glyph.meta_context["preset_applied"] = preset
        glyph.meta_context["preset_description"] = preset_data["description"]
        print(f"INFO: Applied preset '{preset}' to glyph: {glyph.glyph_id}")
        return glyph

    def get_available_presets(self) -> List[str]:
        """
        Returns the list of available presets.

        Returns:
            List of preset names.
        """
        return list(self.presets.keys())