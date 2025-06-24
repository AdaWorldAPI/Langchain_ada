"""
Felt Scene – Models Open-World Qualia
------------------------------------
Models qualia scenes with FeltDTOs, presets, and keyframes for cinematic rendering.
"""

from typing import List, Dict
from felt_dto_v5 import FeltDTO

class Keyframe:
    """Represents a single keyframe in a felt scene."""
    def __init__(self, timestamp: float, glyph: FeltDTO):
        self.timestamp = timestamp
        self.glyph = glyph

class FeltScene:
    """Models a qualia-driven scene with keyframes."""
    def __init__(self, keyframes: List[Keyframe] = None):
        self.keyframes = keyframes or []

    def render(self) -> Dict:
        """Renders the scene as a sequence of glyphs."""
        return {"keyframes": [{"timestamp": kf.timestamp, "glyph": kf.glyph.to_dict()} for kf in self.keyframes]}