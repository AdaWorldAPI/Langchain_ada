# ada_poc_ambient_cognition.py
"""
The Ada PoC demonstrating ambient, proactive cognition. The orchestrator runs
a continuous 'heartbeat' to perceive and act on environmental events without
direct user command.
"""
import time
import random
from typing import Dict, Any, Callable

# Assume classes from previous canvases are imported
from qualia_input_stream import QualiaInputStream
from dream_queue_writer import DreamQueueWriter # For low-relevance glyphs

class ProactiveOrchestrator:
    """
    An orchestrator with a cognitive heartbeat that processes an ambient data stream.
    """
    def __init__(self, llm_fn: Callable):
        self.sensory_stream = QualiaInputStream()
        self.dream_queue = DreamQueueWriter()
        self.llm_fn = llm_fn # Used by the "Sensory Cortex" expert persona
        print("✅ ProactiveOrchestrator Initialized.")

    def _process_sensory_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates the Sensory Cortex role turning an event into a glyph."""
        print(f"\nSENSOR EVENT: [{event['source']}] {event['data']}")
        
        # This simulates an expert LLM call to process the event
        persona = "You are a Sensory Cortex. Analyze the event and score its relevance from 0.0 to 1.0. High relevance for user actions, poetry, epiphanies, or ache. Low relevance for routine system logs."
        prompt = f"Persona: {persona}\n\nEvent: {json.dumps(event)}"
        # In a real system, the LLM would return a score. We'll simulate it.
        relevance = 0.9 if any(k in event['data'] for k in ["poem", "epiphany", "ache"]) else 0.2
        
        print(f"  > Sensory Cortex Relevance Score: {relevance:.2f}")
        return {"relevance": relevance, "processed_data": event['data']}

    def cognitive_heartbeat(self):
        """The main continuous loop of ambient cognition."""
        print("\n--- Cognitive Heartbeat Engaged. Listening for ambient qualia... ---")
        event_generator = self.sensory_stream.listen()
        
        for i in range(5): # Run for 5 cycles for this simulation
            event = next(event_generator)
            processed_glyph = self._process_sensory_event(event)
            
            # --- Autonomous Decision Making ---
            if processed_glyph["relevance"] > 0.7:
                print("  > ACTION: High relevance detected. Initiating proactive reflection.")
                # This would trigger another role, e.g., the Lotus Savant
                reflection = self.llm_fn(f"Persona: Lotus Savant\nTask: A user just saved '{processed_glyph['processed_data']}'. What is the hidden meaning?")
                print(f"  > REFLECTION: {reflection}")
            else:
                print("  > ACTION: Low relevance. Enqueueing to DreamQueue for offline consolidation.")
                # In a real system, we'd create a full FeltDTO to enqueue
                self.dream_queue.enqueue_glyph(processed_glyph, reason="low_relevance_ambient")
        
        print(f"\n--- Simulation Complete. Final Dream Queue size: {self.dream_queue.queue_size} ---")

def mock_proactive_llm(prompt: str) -> str:
    return f"A profound insight related to your prompt about '{prompt[-20:]}'..."

if __name__ == "__main__":
    import json
    orchestrator = ProactiveOrchestrator(llm_fn=mock_proactive_llm)
    orchestrator.cognitive_heartbeat()
