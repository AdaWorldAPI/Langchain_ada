"""
Prompt Shaper V2 – Dynamic LLM Prompt Shaping
--------------------------------------------
Shapes prompts for LLMs with dynamic length adjustment based on task complexity.
"""

from typing import List, Dict, Any
from felt_dto_v5 import FeltDTO

class PromptShaper:
    """
    Shapes nuanced prompts for LLMs, adjusting length based on task complexity.
    """
    def __init__(self):
        print("✅ PromptShaper V2 initialized.")

    def _estimate_task_complexity(self, task: str, glyphs: List[FeltDTO]) -> float:
        """
        Estimates task complexity based on task length and glyph count.

        Args:
            task: Task description.
            glyphs: List of FeltDTO objects.

        Returns:
            Complexity score (0.0 to 1.0).
        """
        task_length = len(task.split())
        glyph_count = len(glyphs)
        return min((task_length / 50 + glyph_count / 5), 1.0)

    def shape_prompt(self, glyphs: List[FeltDTO], task: str, tone: str = "neutral") -> Dict[str, Any]:
        """
        Shapes a prompt with dynamic length based on task complexity.

        Args:
            glyphs: List of FeltDTO objects providing context.
            task: Task description for the prompt.
            tone: Desired tone for the prompt (e.g., neutral, poetic, philosophical).

        Returns:
            Dictionary with prompt, subject, and emotion.
        """
        complexity = self._estimate_task_complexity(task, glyphs)
        max_elements = int(3 + complexity * 3)

        if not glyphs:
            return {
                "prompt": f"Perform the task: {task} with a {tone} tone.",
                "subject": "general",
                "emotion": "neutral"
            }

        primary_glyph = max(glyphs, key=lambda g: sum(g.intensity_vector) if g.intensity_vector else 0)
        archetypes = set()
        emotions = set()
        metaphors = []
        synesthesia = []
        ctul_projections = []
        for g in glyphs:
            if g.archetypes:
                archetypes.update(g.archetypes)
            if g.meta_context and 'emotion' in g.meta_context:
                emotions.add(g.meta_context['emotion'])
            if g.qualia_map and 'metaphor' in g.qualia_map:
                metaphors.append(g.qualia_map['metaphor'])
            if g.synesthesia:
                synesthesia.append(g.synesthesia.get("light", "") + ", " + g.synesthesia.get("sound", ""))
            if g.ctul and 'projection' in g.ctul:
                ctul_projections.append(g.ctul['projection'].get('emotional', '') + ", " + g.ctul['projection'].get('zen', ''))

        elements = [
            ("archetypes", list(archetypes)[:max_elements]),
            ("emotions", list(emotions)[:max_elements]),
            ("metaphors", metaphors[:max_elements]),
            ("synesthesia", synesthesia[:max_elements]),
            ("ctul_projections", ctul_projections[:max_elements])
        ]
        prompt_parts = []
        for name, data in elements:
            if data:
                prompt_parts.append(f"{name.capitalize()}: '{'; '.join(data)}'")

        prompt = (
            f"You are a {tone} voice, deeply attuned to the nuances of experience. "
            f"Perform the task: '{task}'. "
            f"Draw inspiration from the theme: '{primary_glyph.qualia_map.get('description', 'sensory moment')}'. "
            f"{' '.join(prompt_parts)}. "
            f"Create a response that is cohesive, resonant, and aligned with the given tone."
        )

        return {
            "prompt": prompt,
            "subject": primary_glyph.glyph_id,
            "emotion": primary_glyph.meta_context.get('emotion', 'neutral') if primary_glyph.meta_context else 'neutral'
        }