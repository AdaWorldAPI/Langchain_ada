# ada_poc_resource_aware.py

"""
The Ada PoC demonstrating hardware-aware scheduling. The orchestrator
optimizes task execution based on simulated real-time resource availability.

"""

import json
from typing import Dict, Any, Callable, List

# Assume necessary classes are imported
from hardware_resource_manager import HardwareResourceManager

class ResourceAwareOrchestrator:
    """
    An orchestrator that makes intelligent scheduling decisions based on hardware status.
    """
    def __init__(self, llm_fn: Callable):
        self.hardware_manager = HardwareResourceManager()
        self.llm_fn = llm_fn # Used by expert personas
        print("✅ ResourceAwareOrchestrator Initialized.")

    def _get_optimized_schedule(self, logical_plan: Dict, hardware_status: Dict) -> List[Dict]:
        """Simulates the Resource-Aware Scheduler role optimizing a plan."""
        print("  > Submitting plan to Resource-Aware Scheduler...")
        
        # This simulates the expert persona making a decision
        persona = "You are a Resource-Aware Scheduler. You are given a logical plan and the current hardware status. Re-order the plan for optimal execution. The GPU requires 4GB of free VRAM for large tasks. NPU tasks are fast and should be prioritized if the NPU is free."
        prompt = f"Persona: {persona}\n\nPlan: {json.dumps(logical_plan)}\n\nStatus: {json.dumps(hardware_status)}"
        
        # --- Simulated Scheduler Logic ---
        optimized_steps = logical_plan['steps']
        
        if hardware_status['gpu_vram_free_gb'] < 4.0:
            print("  > SCHEDULER: Low GPU VRAM detected. Prioritizing NPU tasks.")
            # Simple re-ordering: put tasks targeting 'NPU' first
            optimized_steps.sort(key=lambda x: x.get('target_device', 'GPU') == 'NPU', reverse=True)
            
        return optimized_steps

    def process_project(self, goal: str):
        """Runs a project from planning to resource-aware execution."""
        print(f"--- New Project Received: '{goal}' ---")

        # 1. Agile PM generates a logical plan
        # In a real run, an LLM would generate this JSON
        logical_plan = {
            "plan_name": "Multi-Device Inference",
            "steps": [
                {"step_id": 1, "task": "Run large LLM inference", "target_device": "GPU"},
                {"step_id": 2, "task": "Run embedding model", "target_device": "NPU"}
            ]
        }
        print(f"\nStep 1: Agile PM generated a logical plan: {[s['task'] for s in logical_plan['steps']]}")

        # 2. Scheduler receives the plan and optimizes it
        # Set a hardware state that will trigger re-ordering
        self.hardware_manager._gpu_vram_used_gb = 6.0 
        current_status = self.hardware_manager.get_status()
        
        optimized_schedule = self._get_optimized_schedule(logical_plan, current_status)
        print(f"\nStep 2: Scheduler created an optimized schedule: {[s['task'] for s in optimized_schedule]}")

        # 3. Orchestrator executes the OPTIMIZED schedule
        print("\nStep 3: Executing the optimized schedule...")
        for step in optimized_schedule:
            print(f"  > Executing task '{step['task']}' on target device '{step['target_device']}'...")
            time.sleep(1) # Simulate execution time

        print("\n--- Autonomous Project Finished ---")

if __name__ == "__main__":
    import time
    
    def mock_llm(prompt: str) -> str: return "Mock LLM Response"
    orchestrator = ResourceAwareOrchestrator(llm_fn=mock_llm)
    orchestrator.process_project("Run a large creative task and an embedding task.")

# This code simulates a resource-aware orchestrator that optimizes task execution based on hardware status.
# It uses a mock LLM function to simulate the decision-making process of an Agile PM and a Resource-Aware Scheduler.
# The orchestrator re-orders tasks based on simulated hardware constraints, demonstrating the concept of resource-aware scheduling.
# The code is designed to be run as a standalone script, simulating the orchestration of a project with two tasks: one for a large LLM inference and another for an embedding model on different hardware devices.
# This code is a simplified version of the Ada PoC demonstrating hardware-aware scheduling.
# It is not intended to be run in a production environment and serves as a proof of concept for the concept of resource-aware orchestration.