"""
Orchestrator NPU V2 – Parallel NPU-Optimized Task Orchestration
-------------------------------------------------------------
Orchestrates task execution with glyph-compressed weights and MiniLM-L6-V2.
"""

from typing import List, Dict, Any, Optional
from service_locator import ServiceLocator
from expert_selector_v20 import ExpertSelector
from felt_dto_v5 import FeltDTO
from openvino_adapter_v1_1 import OpenVINOAdapter
from hardware_resource_manager import HardwareResourceManager
from deliberate_pause_agent_v2 import DeliberatePauseAgent
from glyph_weight_compressor import GlyphWeightCompressor
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class OrchestratorNPU:
    """
    NPU-optimized orchestrator with parallel execution and glyph-compressed weights.
    """
    def __init__(self, locator: ServiceLocator = None, max_workers: int = 4):
        """
        Initializes the OrchestratorNPU.

        Args:
            locator: ServiceLocator instance.
            max_workers: Maximum number of parallel workers.
        """
        self.locator = locator or ServiceLocator()
        self.selector = ExpertSelector(self.locator)
        self.pause_agent = self.locator.get("DeliberatePauseAgent") or DeliberatePauseAgent()
        self.hardware_manager = self.locator.get("hardware_manager") or HardwareResourceManager()
        self.openvino_adapter = self.locator.get("openvino_adapter") or OpenVINOAdapter()
        self.compressor = GlyphWeightCompressor()
        self.max_workers = max_workers

        if not self.hardware_manager.allocate("npu", 0.8):
            print("⚠️ Falling back to CPU due to NPU unavailability.")
            self.hardware_manager.allocate("cpu", 0.8)
        
        print("✅ OrchestratorNPU V2 initialized with MiniLM-L6-V2.")

    def invoke_chain(self, task: str, chain: Optional[List[str]] = None, event: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Invokes a chain of experts with glyph-compressed weights.

        Args:
            task: Task description.
            chain: Optional list of expert names.
            event: Event dictionary with source and data.

        Returns:
            Dictionary with execution results.
        """
        context = {"initial_prompt": task, "event": event or {}}
        glyphs = [FeltDTO.from_dict(data) for data in event.get("data", [{}])] if event and isinstance(event.get("data"), list) else [FeltDTO.from_dict(event.get("data", {}))]

        if not chain:
            chain = self.selector.classify_task(glyph=glyphs[0], query=task)
            print(f"--- Inferred Chain: {chain} ---")
        
        optimized_chain = self._optimize_chain(chain, context)
        print(f"Optimized Chain: {optimized_chain}")

        for step, expert_name in enumerate(optimized_chain, 1):
            print(f"  > Step {step}: Passing context to '{expert_name}'...")
            
            contexts = [context.copy() for _ in glyphs]
            results = self._parallel_execute(expert_name, contexts, glyphs)

            context = results[0] if results else context
            for i, result in enumerate(results[1:], 1):
                context[f"output_glyph_{i}"] = result.get("creative_output", result.get("output", ""))
            
            context[f"pause_decision_{expert_name}"] = str(context.get(expert_name, context.get("creative_output", context.get("output", ""))))
            context = self.pause_agent.reflect(context)

        print("--- Chain execution complete. ---")
        return context

    def _parallel_execute(self, expert_name: str, contexts: List[Dict], glyphs: List[FeltDTO]) -> List[Dict]:
        """
        Executes an expert in parallel for multiple glyphs.

        Args:
            expert_name: Name of the expert.
            contexts: List of context dictionaries.
            glyphs: List of FeltDTO objects.

        Returns:
            List of updated context dictionaries.
        """
        results = []
        expert = self.selector.select_expert(expert_name, contexts[0])

        if not expert:
            print(f"⚠️ Expert '{expert_name}' not found.")
            return contexts

        def execute_single(context: Dict, glyph: FeltDTO) -> Dict:
            if self.hardware_manager.allocate("npu", 0.2) or self.hardware_manager.allocate("cpu", 0.2):
                start_time = time.time()
                weight_glyphs = self.compressor.load_glyphs(glyph.vector_embedding, k=10)
                context["weight_glyphs"] = [g.to_dict() for g in weight_glyphs]
                context = expert.invoke(context, glyph)
                elapsed_time = time.time() - start_time
                print(f"    > Executed '{expert_name}' for glyph {glyph.glyph_id} in {elapsed_time:.3f} seconds.")
                self.hardware_manager.release("npu", 0.2) or self.hardware_manager.release("cpu", 0.2)
            else:
                context["output"] = f"Fallback: No resources for glyph {glyph.glyph_id}"
            return context

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_context = {executor.submit(execute_single, ctx, g): ctx for ctx, g in zip(contexts, glyphs)}
            for future in as_completed(future_to_context):
                results.append(future.result())
        
        return results

    def __del__(self):
        """
        Releases allocated hardware resources.
        """
        self.hardware_manager.release("npu", 0.8) or self.hardware_manager.release("cpu", 0.8)
        print("✅ OrchestratorNPU resources released.")