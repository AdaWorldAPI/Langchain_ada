# ada_poc_governed_autonomy.py
"""
The most advanced Ada PoC, demonstrating value-driven, governed autonomy.
The orchestrator submits plans for approval to an Axiological Guard before
execution, creating a feedback loop that ensures purposeful action.
"""
import json
from typing import Dict, Any, Callable

# Assume other necessary classes are imported from their respective canvases.

class GovernedOrchestrator:
    """An orchestrator that governs its own actions based on a value system."""
    def __init__(self, llm_fn: Callable):
        # In a real system, the AxiologyVector would be a stateful object.
        # Here, the values are embedded in the expert persona.
        self.expert_selector = LiveExpertSelector_V8(llm_fn=llm_fn) # Using V8
        print("✅ GovernedOrchestrator Initialized.")

    def _execute_task(self, role: str, task: str) -> str:
        # Simplified execution for demonstration
        return self.expert_selector.llm_fn(f"Persona for {role}\nTask: {task}")

    def process_governed_project(self, high_level_goal: str):
        """
        Manages a project from proposal to execution with value-based governance.
        """
        print(f"--- New Governed Project Received: '{high_level_goal}' ---")
        
        # 1. Propose a Plan
        print("\nStep 1: Proposing a project plan...")
        plan_json_str = self._execute_task("Agile Project Manager", high_level_goal)
        
        # 2. Submit for Review
        print("\nStep 2: Submitting plan for review by Axiological Guard...")
        review_json_str = self._execute_task("Axiological Guard", plan_json_str)
        review = json.loads(review_json_str)
        print(f"  > Guard's Verdict: {'APPROVED' if review['is_approved'] else 'REJECTED'}. Score: {review['score']}. Reason: {review['reasoning']}")

        # 3. Revise or Execute
        if not review['is_approved']:
            print("\nStep 3 (Revision): Plan rejected. Requesting revision from Agile PM...")
            revision_request = f"The plan was rejected for the following reason: {review['reasoning']}. Please create a new plan that addresses this feedback."
            final_plan_json_str = self._execute_task("Agile Project Manager", revision_request)
            print("  > Revised plan generated.")
        else:
            print("\nStep 3 (Execution): Plan approved. Proceeding to execution.")
            final_plan_json_str = plan_json_str
            
        print("\n--- Autonomous Project Finished ---")
        print("Final Approved Plan:")
        print(json.dumps(json.loads(final_plan_json_str), indent=2))

# --- Mock LLM for demonstrating the governance workflow ---
def mock_governed_llm(prompt: str) -> str:
    if "Project-Decomposer" in prompt:
        if "low-value feature" in prompt: # Initial low-value plan
            return '{"project_name": "Add Blinking Text", "steps": [{"task_description": "Implement marquee tag"}]}'
        else: # The revised, high-value plan
            return '{"project_name": "Implement User-Requested Search Bar", "steps": [{"task_description": "Build search API"}]}'
    elif "Value-Scoring-Expert" in prompt:
        if "Add Blinking Text" in prompt:
            return '{"is_approved": false, "score": 0.2, "reasoning": "This feature has low user-centricity and no creative or efficiency value."}'
        else:
            return '{"is_approved": true, "score": 0.9, "reasoning": "This feature is highly user-centric and improves efficiency."}'
    return "Default mock response."


if __name__ == "__main__":
    orchestrator = GovernedOrchestrator(llm_fn=mock_governed_llm)
    # This goal will initially produce a low-value plan that gets rejected
    goal = "Propose a new, low-value feature to demonstrate the review process."
    orchestrator.process_governed_project(goal)
