"""
Orchestrator NPU V1 – NPU-Optimized Task Orchestration
-----------------------------------------------------
Optimizes LangChain-inspired task orchestration for NUC 14 Pro with NPU, using OpenVINO and Soulframe Engine.
"""

from typing import List, Dict, Any, Optional
from service_locator import ServiceLocator
from expert_selector_v20 import ExpertSelector
from felt_dto_v5 import FeltDTO
from openvino_adapter_v1_1 import OpenVINOAdapter
from hardware_resource_manager import HardwareResourceManager
from deliberate_pause_agent_v2 import DeliberatePauseAgent
import numpy as np
import time

class OrchestratorNPU:
    """
    NPU-optimized orchestrator for task execution, integrating LangChain-inspired chaining with OpenVINO acceleration.
    """
    def __init__(self, locator: ServiceLocator = None):
        """
        Initializes the OrchestratorNPU.

        Args:
            locator: ServiceLocator instance for dependency injection.
        """
        self.locator = locator or ServiceLocator()
        self.selector = ExpertSelector(self.locator)
        self.pause_agent = self.locator.get("DeliberatePauseAgent") or DeliberatePauseAgent()
        self.hardware_manager = self.locator.get("hardware_manager") or HardwareResourceManager()
        self.openvino_adapter = self.locator.get("openvino_adapter") or OpenVINOAdapter()
        
        # Allocate NPU resources
        if not self.hardware_manager.allocate("npu", 0.8):
            print("⚠️ Falling back to CPU due to NPU unavailability.")
            self.hardware_manager.allocate("cpu", 0.8)
        
        print("✅ OrchestratorNPU V1 initialized with NPU acceleration.")

    def invoke_chain(self, task: str, chain: Optional[List[str]] = None, event: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Invokes a chain of experts for a task, optimized for NPU.

        Args:
            task: Task description to execute.
            chain: Optional list of expert names; if None, inferred by selector.
            event: Event dictionary containing source and data (e.g., FeltDTO).

        Returns:
            Dictionary with execution results.
        """
        context = {"initial_prompt": task, "event": event or {}}
        glyph = FeltDTO.from_dict(event.get("data", {})) if event and "data" in event else FeltDTO()

        # Infer chain if not provided
        if not chain:
            chain = self.selector.classify_task(glyph=glyph, query=task)
            print(f"--- Inferred Chain: {chain} ---")
        
        # Optimize chain for NPU
        optimized_chain = self._optimize_chain(chain, context)
        print(f"Optimized Chain: {optimized_chain}")

        # Execute chain
        for step, expert_name in enumerate(optimized_chain, 1):
            print(f"  > Step {step}: Passing context to '{expert_name}'...")
            
            # Request NPU resources for expert execution
            if not self.hardware_manager.allocate("npu", 0.2):
                print("    > NPU busy, using CPU fallback.")
                self.hardware_manager.allocate("cpu", 0.2)

            expert = self.selector.select_expert(expert_name, context)
            if expert:
                start_time = time.time()
                context = expert.invoke(context, glyph)
                context[f"pause_decision_{expert_name}"] = str(context.get(expert_name, context.get("creative_output", context.get("output", ""))))
                
                # Reflect with pause
                context = self.pause_agent.reflect(context)
                elapsed_time = time.time() - start_time
                print(f"    > Executed '{expert_name}' in {elapsed_time:.3f} seconds.")

                # Release resources
                self.hardware_manager.release("npu", 0.2) or self.hardware_manager.release("cpu", 0.2)
            else:
                print(f"⚠️ Expert '{expert_name}' not found.")

        print("--- Chain execution complete. ---")
        return context

    def _optimize_chain(self, chain: List[str], context: Dict[str, Any]) -> List[str]:
        """
        Optimizes the expert chain for NPU execution based on task complexity and resource availability.

        Args:
            chain: Initial expert chain.
            context: Context dictionary for optimization.

        Returns:
            Optimized expert chain.
        """
        # Mock optimization; replace with NPU-specific heuristics
        task = context.get("initial_prompt", "")
        complexity = len(task.split()) / 50.0  # Simple heuristic
        if complexity < 0.5 and "prompt_shaper" in chain:
            # Skip heavy preprocessing for simple tasks
            return [exp for exp in chain if exp != "prompt_shaper"]
        return chain

    def __del__(self):
        """
        Releases allocated hardware resources on destruction.
        """
        self.hardware_manager.release("npu", 0.8) or self.hardware_manager.release("cpu", 0.8)
        print("✅ OrchestratorNPU resources released.")

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    locator = ServiceLocator()
    locator.register("hardware_manager", HardwareResourceManager())
    locator.register("openvino_adapter", OpenVINOAdapter())
    orchestrator = OrchestratorNPU(locator)
    glyph = FeltDTO(
        glyph_id="test_glyph",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"}
    )
    result = orchestrator.invoke_chain(
        task="Generate a poetic reflection",
        event={"source": "test", "data": glyph.to_dict()}
    )
    print(result)