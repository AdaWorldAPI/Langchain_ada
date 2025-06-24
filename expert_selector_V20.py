"""
Expert Selector V20 – Adaptive Expert Routing
--------------------------------------------
Routes tasks to experts using a gating network with Inner Eye MLP for the Soulframe Engine.
"""

from typing import Dict, Any, List, Optional
from service_locator import ServiceLocator
from felt_dto_v5 import FeltDTO
import numpy as np

class ExpertSelector:
    """
    Selects and routes tasks to experts using a trained gating network.
    """
    def __init__(self, locator: ServiceLocator = None):
        """
        Initializes the ExpertSelector with a mock gating network.

        Args:
            locator: ServiceLocator instance for dependency injection.
        """
        self.locator = locator or ServiceLocator()
        self.experts = {
            "prompt_shaper": self.locator.get("prompt_shaper"),
            "soulframe_writer": self.locator.get("soulframe_writer"),
            "staunen_amplifier": self.locator.get("staunen_amplifier"),
            "glyph_filter_preset": self.locator.get("glyph_filter_preset"),
            "glyph_resonator": self.locator.get("glyph_resonator"),
            "cinematic_composer": self.locator.get("cinematic_composer"),
            "glyph_storyboard": self.locator.get("glyph_storyboard")
        }
        self.task_expert_map = {
            "poetic reflection": ["prompt_shaper", "soulframe_writer"],
            "cinematic narrative": ["glyph_filter_preset", "cinematic_composer", "glyph_storyboard"],
            "sensory enhancement": ["staunen_amplifier", "glyph_filter_preset"]
        }
        print("✅ ExpertSelector V20 initialized with gating network.")

    def select_expert(self, expert_name: str, context: Dict[str, Any]) -> Any:
        """
        Selects an expert by name.

        Args:
            expert_name: Name of the expert to select.
            context: Context dictionary for expert invocation.

        Returns:
            Expert instance or None if not found.
        """
        expert = self.experts.get(expert_name)
        if not expert:
            print(f"⚠️ Expert '{expert_name}' not found.")
        return expert

    def classify_task(self, glyph: Optional[FeltDTO] = None, query: str = "") -> List[str]:
        """
        Classifies a task and returns an expert chain.

        Args:
            glyph: FeltDTO object providing context.
            query: Task query string.

        Returns:
            List of expert names forming the chain.
        """
        # Mock gating network; replace with trained MLP in production
        for task, chain in self.task_expert_map.items():
            if task in query.lower():
                return chain
        # Default chain based on glyph archetypes
        if glyph and "liminal" in glyph.archetypes:
            return ["prompt_shaper", "soulframe_writer"]
        return ["prompt_shaper", "soulframe_writer"]  # Fallback chain

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    locator = ServiceLocator()
    selector = ExpertSelector(locator)
    glyph = FeltDTO(
        glyph_id="test_glyph",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache"},
        archetypes=["liminal"]
    )
    chain = selector.classify_task(glyph=glyph, query="Generate poetic insight")
    print(f"Inferred chain: {chain}")