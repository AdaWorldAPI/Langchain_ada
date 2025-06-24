"""
Ada PoC Environmental Interaction – Orchestrates Environmental Inputs
--------------------------------------------------------------------
Orchestrates interactions between Ada’s cognitive architecture and environmental stimuli.
"""

from typing import Dict, Any, List
from felt_dto_v5 import FeltDTO
from service_locator import ServiceLocator
from langchain_inspired_orchestrator_v6 import LangChainInspiredOrchestrator
from archetype_mapper import ArchetypeMapper
import numpy as np

class EnvironmentalInteractionOrchestrator:
    """
    Orchestrates environmental inputs as FeltDTO glyphs, driving sensory and narrative responses.
    """
    def __init__(self, locator: ServiceLocator = None):
        """
        Initializes the EnvironmentalInteractionOrchestrator.

        Args:
            locator: ServiceLocator instance for dependency injection.
        """
        self.locator = locator or ServiceLocator()
        self.orchestrator = self.locator.get("orchestrator") or LangChainInspiredOrchestrator()
        self.mapper = ArchetypeMapper()
        print("✅ EnvironmentalInteractionOrchestrator initialized.")

    def process_environmental_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes environmental input data into a FeltDTO glyph and orchestrates a response.

        Args:
            input_data: Dictionary containing environmental stimuli (e.g., sensor data, user input).

        Returns:
            Dictionary with processed output and glyph.
        """
        # Convert input data to FeltDTO
        glyph = self._create_glyph_from_input(input_data)
        
        # Map archetypes to the glyph
        self.mapper.map_archetypes(glyph)

        # Define orchestration chain
        chain = ["staunen_amplifier", "glyph_filter_preset", "prompt_shaper", "soulframe_writer"]

        # Execute chain
        context = {
            "glyph": glyph,
            "task": input_data.get("task", "Generate a poetic reflection"),
            "additional_glyphs": []
        }
        result = self.orchestrator.invoke_chain(
            context["task"],
            chain,
            event={"source": "environment", "data": glyph.to_dict()}
        )

        result["glyph"] = glyph.to_dict()
        print(f"INFO: Processed environmental input for glyph: {glyph.glyph_id}")
        return result

    def _create_glyph_from_input(self, input_data: Dict[str, Any]) -> FeltDTO:
        """
        Creates a FeltDTO glyph from environmental input data.

        Args:
            input_data: Dictionary containing environmental stimuli.

        Returns:
            FeltDTO object representing the input.
        """
        stimulus = input_data.get("stimulus", "unknown")
        emotion = input_data.get("emotion", "neutral")
        intensity = input_data.get("intensity", [0.5, 0.5, 0.5, 0.5])

        return FeltDTO(
            glyph_id=f"env_{np.random.randint(1000, 9999)}",
            intensity_vector=intensity,
            meta_context={"emotion": emotion, "source": "environment", "stimulus": stimulus},
            qualia_map={"description": stimulus},
            archetypes=["ambient", "sensory"],
            vector_embedding=np.random.rand(384).astype(np.float32),
            staunen_markers=[50, 50, 50, 50]
        )

if __name__ == "__main__":
    from service_locator import ServiceLocator
    locator = ServiceLocator()
    orchestrator = EnvironmentalInteractionOrchestrator(locator=locator)
    input_data = {
        "stimulus": "A gentle breeze",
        "emotion": "calm",
        "intensity": [0.3, 0.4, 0.5, 0.2],
        "task": "Generate a poetic reflection"
    }
    result = orchestrator.process_environmental_input(input_data)
    print(result)