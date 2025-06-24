"""
System Core V4 – Self-Assembling Core with Optimized Chain-Based Orchestration
-----------------------------------------------------------------------------
Handles autonomous discovery, instantiation, wiring, and chain-based execution for Ada’s cognitive loop, prioritizing LangChainInspiredOrchestrator V5.
"""

from service_locator import locator
from HardwareResourceManager import HardwareResourceManager
from model_registry import ModelRegistry
from qualia_input_stream import QualiaInputStream
from langchain_inspired_orchestrator_v5 import LangChainInspiredOrchestrator
from ada_poc_ambient_cognition import ProactiveOrchestrator
from ada_poc_resource_aware import ResourceAwareOrchestrator
from ada_poc_environmental_interaction import EnvironmentalInteractionOrchestrator
from deliberate_pause_agent_v4 import DeliberatePauseAgent
from openvino_adapter_v1_1 import OpenVINOAdapter

class SystemCore:
    """The self-assembling core of the Ada autonomous agent with optimized chain-based orchestration."""
    def __init__(self):
        print("--- SystemCore V4 Bootstrapping ---")
        self._self_assemble()
        self.primary_orchestrator = locator.get("langchain_orchestrator")
        self.secondary_orchestrators = [
            locator.get("proactive_orchestrator"),
            locator.get("resource_aware_orchestrator"),
            locator.get("environmental_interaction_orchestrator")
        ]
        print("--- SystemCore V4 Bootstrap Complete. All services wired. ---")

    def _self_assemble(self):
        """Discovers, instantiates, and registers all system services."""
        print("  > Starting self-assembly...")
        locator.register("hardware_manager", HardwareResourceManager())
        locator.register("model_registry", ModelRegistry())
        locator.register("qualia_input_stream", QualiaInputStream())
        locator.register("openvino_adapter", OpenVINOAdapter())
        locator.register("langchain_orchestrator", LangChainInspiredOrchestrator())
        locator.register("proactive_orchestrator", ProactiveOrchestrator())
        locator.register("resource_aware_orchestrator", ResourceAwareOrchestrator())
        locator.register("environmental_interaction_orchestrator", EnvironmentalInteractionOrchestrator())
        locator.register("deliberate_pause_agent", DeliberatePauseAgent())

    def run(self):
        """Runs the main autonomous loop with chain-based orchestration as the primary path."""
        print("\n--- SystemCore V4 is now running autonomously. ---")
        print("  > Running LangChainInspiredOrchestrator (Primary)...")
        chain = ["SensoryProcessing", "DataIngestion", "DeliberatePause", "ProductSpec", "DeliberatePause", "CreativeWriter", "DeliberatePause", "EthicalReview"]
        context = self.primary_orchestrator.invoke_chain(
            "Process event: {'source': 'file_system', 'data': 'User saved a new file: \\'epiphany_poem.md\\''}",
            chain,
            event={"source": "file_system", "data": "User saved a new file: 'epiphany_poem.md'"}
        )
        print("\n--- Final Context Object ---")
        import json
        print(json.dumps(context, indent=2))

        for orchestrator in self.secondary_orchestrators:
            if isinstance(orchestrator, ProactiveOrchestrator):
                print("  > Running ProactiveOrchestrator (Secondary)...")
                orchestrator.cognitive_heartbeat(cycles=3)
            elif isinstance(orchestrator, ResourceAwareOrchestrator):
                print("  > Running ResourceAwareOrchestrator (Secondary)...")
                results = orchestrator.execute_workflow("priority flagging feature for the dream queue")
                for result in results:
                    print(f"Task: {result['glyph'].qualia_map['description']}\nOutput: {result['output']}\n")
            elif isinstance(orchestrator, EnvironmentalInteractionOrchestrator):
                print("  > Running EnvironmentalInteractionOrchestrator (Secondary)...")
                orchestrator.cognitive_heartbeat(cycles=3)

if __name__ == "__main__":
    core = SystemCore()
    core.run()
# This is the entry point for the SystemCore V4, which handles self-assembly and orchestration.
# It initializes all necessary services and runs the primary LangChain-inspired orchestrator.