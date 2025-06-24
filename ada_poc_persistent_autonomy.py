# ada_poc_persistent_autonomy.py
"""
The most advanced Ada PoC, demonstrating stateful, persistent autonomy.
The orchestrator can start, interrupt, and resume multi-step projects.
"""

import json
from typing import Dict, Any, Callable
import uuid

# Assume necessary classes are imported from their respective canvases.
# For this example, we mock them.

class MockLLM:
    def __init__(self):
        self.plan_generated = False

    def get_function(self):
        def _llm(prompt: str) -> str:
            if "Project-Decomposer-V3" in prompt:
                project_id = f"proj_{uuid.uuid4().hex[:8]}"
                return f'''
                {{
                  "project_id": "{project_id}",
                  "project_name": "End-to-End Feature: Priority Flagging",
                  "status": "in-progress",
                  "steps": [
                    {{
                      "step_id": 1, "task_description": "Define user stories.", "assigned_role": "Lead Product Engineer", "output_variable": "user_stories", "status": "pending", "output_artifact": null
                    }},
                    {{
                      "step_id": 2, "task_description": "Generate tests for {{user_stories}}.", "assigned_role": "Visual Studio Test VM", "output_variable": "test_script", "status": "pending", "output_artifact": null
                    }}
                  ]
                }}
                '''
            elif "User-Story-Writer" in prompt:
                return "User Story: As a system, I want to flag a glyph as 'high priority'."
            elif "Pytest-Generator" in prompt:
                return "# pytest script generated based on context..."
            return "Default mock response."
        return _llm

class StatefulOrchestrator:
    """An orchestrator that manages project state."""
    def __init__(self, llm_fn: Callable):
        # The 'Project Memory' simulates a persistent database (e.g., Redis, MongoDB)
        self.project_memory: Dict[str, Dict[str, Any]] = {}
        self.expert_selector_llm = llm_fn
        print("✅ StatefulOrchestrator Initialized with Project Memory.")

    def _execute_step(self, task_description: str, role: str) -> str:
        # Simplified execution for demonstration
        print(f"  > Executing task for role '{role}': '{task_description[:40]}...'")
        return self.expert_selector_llm(f"Persona for {role}\nTask: {task_description}")

    def start_project(self, high_level_goal: str) -> str:
        """Starts a new project and returns its ID."""
        print(f"--- Starting New Project: '{high_level_goal}' ---")
        plan_json_str = self._execute_step(high_level_goal, "Agile Project Manager")
        plan = json.loads(plan_json_str)
        project_id = plan['project_id']
        self.project_memory[project_id] = plan
        print(f"  > Project plan created and saved with ID: {project_id}")
        return project_id

    def resume_project(self, project_id: str):
        """Resumes a project, executing the next pending step."""
        if project_id not in self.project_memory:
            print(f"❌ ERROR: Project with ID '{project_id}' not found.")
            return

        project = self.project_memory[project_id]
        print(f"\n--- Resuming Project: '{project['project_name']}' (ID: {project_id}) ---")

        # Find the next pending step
        next_step = next((s for s in project['steps'] if s['status'] == 'pending'), None)

        if not next_step:
            project['status'] = 'completed'
            print("  > All steps completed. Project is now finished.")
            return
        
        # Build context from previously completed steps
        context = {s['output_variable']: s['output_artifact'] for s in project['steps'] if s['status'] == 'completed'}
        
        # Execute the step
        task_desc = next_step['task_description'].format(**context)
        step_output = self._execute_step(task_desc, next_step['assigned_role'])

        # Update the project state
        next_step['output_artifact'] = step_output
        next_step['status'] = 'completed'
        self.project_memory[project_id] = project # Save the updated state
        
        print(f"  > Step {next_step['step_id']} completed. Project state saved.")
        # Check if project is now complete
        if not any(s['status'] == 'pending' for s in project['steps']):
            project['status'] = 'completed'
            print("  > All steps completed. Project is now finished.")

if __name__ == "__main__":
    mock_llm = MockLLM()
    orchestrator = StatefulOrchestrator(llm_fn=mock_llm.get_function())
    
    # 1. Start a project. It creates a plan and saves it.
    project_id = orchestrator.start_project(
        "Manage the end-to-end creation of a 'priority flagging' feature."
    )
    
    # 2. Run the first step of the project.
    orchestrator.resume_project(project_id)
    
    print("\n... (System is interrupted or pauses) ...\n")
    
    # 3. Resume the SAME project. It will automatically find and run the next pending step.
    orchestrator.resume_project(project_id)
    
    # 4. Verify completion
    final_state = orchestrator.project_memory[project_id]
    print(f"\nFinal Project Status: {final_state['status']}")
    assert final_state['status'] == 'completed'

