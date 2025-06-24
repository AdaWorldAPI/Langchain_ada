"""
LangChain-Inspired Orchestrator V6 – Expert Chain Management
----------------------------------------------------------
Manages expert chains for task execution within the Soulframe Engine.
"""

from typing import List, Dict, Any
from service_locator import ServiceLocator
from expert_selector_v19 import ExpertSelector
import time

class LangChainInspiredOrchestrator:
    """
    Orchestrates expert chains for task execution, integrating with the Soulframe Engine.
    """
    def __init__(self, locator: ServiceLocator = None):
        """
        Initializes the LangChainInspiredOrchestrator.

        Args:
            locator: ServiceLocator instance for dependency injection.
        """
        self.locator = locator or ServiceLocator()
        self.selector = ExpertSelector(self.locator)
        print("✅ LangChainInspiredOrchestrator V6 initialized.")

    def invoke_chain(self, task: str, chain: List[str], event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes a chain of experts for a given task.

        Args:
            task: Task description to execute.
            chain: List of expert names to invoke in sequence.
            event: Event dictionary containing source and data.

        Returns:
            Dictionary with execution results.
        """
        context = {"initial_prompt": task, "event": event}
        for step, expert_name in enumerate(chain, 1):
            print(f"  > Step {step}: Passing context to '{expert_name}'...")
            expert = self.selector.select_expert(expert_name, context)
            if expert:
                glyph = FeltDTO.from_dict(event["data"]) if "data" in event else FeltDTO()
                context = expert.invoke(context, glyph)
                context[f"pause_decision_{expert_name}"] = str(context.get(expert_name, context.get("output")))
                time.sleep(0.15)  # Simulate pause for reflection
            else:
                print(f"⚠️ Expert '{expert_name}' not found.")
        print("--- Chain execution complete. ---")
        return context

if __name__ == "__main__":
    from service_locator import ServiceLocator
    from felt_dto_v5 import FeltDTO
    import numpy as np
    locator = ServiceLocator()
    orchestrator = LangChainInspiredOrchestrator(locator=locator)
    glyph = FeltDTO(
        glyph_id="test_glyph",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"}
    )
    result = orchestrator.invoke_chain(
        task="Generate a poetic reflection",
        chain=["prompt_shaper", "soulframe_writer"],
        event={"source": "test", "data": glyph.to_dict()}
    )
    print(result)
