"""
Ada PoC Self-Assembling V5 – Dynamic Architecture Assembly
---------------------------------------------------------
Dynamically configures expert chains based on runtime conditions for Ada’s architecture.
"""

from typing import Dict, Any, List, Optional
from service_locator import ServiceLocator
from orchestrator_npu_v2 import OrchestratorNPU
from felt_dto_v5 import FeltDTO
from hardware_resource_manager import HardwareResourceManager
from expert_selector_v20 import ExpertSelector
import numpy as np

class AdaPoCSelfAssembling:
    """
    Self-assembling PoC for Ada’s architecture, compatible with MiniLM-L6-V2.
    """
    def __init__(self, locator: ServiceLocator = None):
        """
        Initializes the self-assembling PoC.

        Args:
            locator: ServiceLocator instance.
        """
        self.locator = locator or ServiceLocator()
        self.hardware_manager = self.locator.get("hardware_manager") or HardwareResourceManager()
        self.orchestrator = self.locator.get("orchestrator") or OrchestratorNPU(self.locator)
        self.selector = self.locator.get("expert_selector") or ExpertSelector(self.locator)
        print("✅ AdaPoCSelfAssembling V5 initialized for MiniLM-L6-V2.")

    def assemble_chain(self, task: str, glyph: FeltDTO, resource_threshold: float = 0.5) -> List[str]:
        """
        Dynamically assembles an expert chain based on task and resources.

        Args:
            task: Task description.
            glyph: FeltDTO object.
            resource_threshold: Minimum available resource fraction.

        Returns:
            List of expert names.
        """
        available_resources = self.hardware_manager.resources.get("npu", {}).get("usage", 1.0)
        if available_resources > resource_threshold:
            return self.selector.classify_task(glyph=glyph, query=task)
        else:
            print("INFO: Limited resources, using lightweight chain.")
            return ["prompt_shaper", "soulframe_writer"]  # Fallback chain

    def run(self, task: str, glyph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task with a dynamically assembled chain.

        Args:
            task: Task description.
            glyph_data: Dictionary containing glyph data.

        Returns:
            Dictionary with execution results.
        """
        glyph = FeltDTO.from_dict(glyph_data)
        chain = self.assemble_chain(task, glyph)
        result = self.orchestrator.invoke_chain(
            task=task,
            chain=chain,
            event={"source": "self_assembling_poc", "data": glyph.to_dict()}
        )
        print(f"INFO: Self-assembling PoC executed task: {task} with chain: {chain}")
        return result

if __name__ == "__main__":
    from service_locator import ServiceLocator
    from hardware_resource_manager import HardwareResourceManager
    import numpy as np
    locator = ServiceLocator()
    locator.register("hardware_manager", HardwareResourceManager())
    poc = AdaPoCSelfAssembling(locator)
    glyph_data = {
        "glyph_id": "hush_touch",
        "intensity_vector": [0.7, 0.6, 0.8, 0.4],
        "meta_context": {"emotion": "ache", "source": "user", "priority": "high"},
        "qualia_map": {"description": "A hush, a glance", "metaphor": "a fleeting shadow"},
        "archetypes": ["liminal", "desire"]
    }
    result = poc.run("Generate a poetic reflection", glyph_data)
    print(result)