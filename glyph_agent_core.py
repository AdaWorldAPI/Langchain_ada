"""
Glyph Agent Core – Core Logic for Glyph Agents
---------------------------------------------
Manages state, context, and interactions for glyph agents within the Soulframe Engine.
"""

from typing import Dict, Any
from felt_dto_v5 import FeltDTO
from service_locator import ServiceLocator

class GlyphAgentCore:
    """
    Core logic for glyph agents, handling state management and task execution.
    """
    def __init__(self, agent_id: str, locator: ServiceLocator):
        """
        Initializes the GlyphAgentCore.

        Args:
            agent_id: Unique identifier for the agent.
            locator: ServiceLocator instance for dependency injection.
        """
        self.agent_id = agent_id
        self.locator = locator
        self.state = {"last_task": None, "last_glyph": None}
        print(f"✅ GlyphAgentCore '{agent_id}' initialized.")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task based on the provided context and glyph.

        Args:
            context: Dictionary containing task, glyph, and agent_id.

        Returns:
            Dictionary with execution results.
        """
        task = context.get("task", "")
        glyph = context.get("glyph", FeltDTO())
        self.state["last_task"] = task
        self.state["last_glyph"] = glyph.glyph_id

        # Delegate to orchestrator for task execution
        orchestrator = self.locator.get("orchestrator")
        if orchestrator:
            chain = ["prompt_shaper", "soulframe_writer"]  # Example chain
            result = orchestrator.invoke_chain(task, chain, event={"source": "agent", "data": glyph.to_dict()})
            context.update(result)
        else:
            context["output"] = f"Mock output for task: {task} with glyph: {glyph.glyph_id}"

        return context

    def update_state(self, state_update: Dict[str, Any]) -> None:
        """
        Updates the agent’s internal state.

        Args:
            state_update: Dictionary containing state updates.
        """
        self.state.update(state_update)

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    locator = ServiceLocator()
    core = GlyphAgentCore(agent_id="agent_001", locator=locator)
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"}
    )
    context = {"task": "Generate a poetic reflection", "glyph": glyph}
    result = core.execute(context)
    print(result)