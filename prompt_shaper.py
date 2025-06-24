"""
Prompt Shaper – Shapes Nuanced LLM Prompts
-----------------------------------------
Shapes prompts for LLMs using FeltDTO context, archetypes, and synesthetic descriptors.
"""

from typing import List, Dict, Any
from felt_dto_v3 import FeltDTO

class PromptShaper:
    """
    Shapes nuanced prompts for LLMs, incorporating emotional and synesthetic context from FeltDTOs.
    """
    def __init__(self):
        print("✅ PromptShaper initialized.")

    def shape_prompt(self, glyphs: List[FeltDTO], task: str, tone: str = "neutral") -> Dict[str, Any]:
        """
        Shapes a prompt based on a list of FeltDTOs and a specified task.

        Args:
            glyphs: List of FeltDTO objects providing context.
            task: Task description for the prompt.
            tone: Desired tone for the prompt (e.g., neutral, poetic, philosophical).

        Returns:
            Dictionary with prompt, subject, and emotion.
        """
        if not glyphs:
            return {
                "prompt": f"Perform the task: {task} with a {tone} tone.",
                "subject": "general",
                "emotion": "neutral"
            }

        # Select the primary glyph based on intensity
        primary_glyph = max(glyphs, key=lambda g: sum(g.intensity_vector) if g.intensity_vector else 0)

        # Aggregate context
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

        # Construct the prompt
        prompt = (
            f"You are a {tone} voice, deeply attuned to the nuances of experience. "
            f"Perform the task: '{task}'. "
            f"Draw inspiration from the theme: '{primary_glyph.qualia_map.get('description', 'sensory moment')}'. "
            f"Resonate with the archetypes: {list(archetypes)}. "
            f"Infuse the emotions: {list(emotions)}. "
            f"Evoke the metaphors: '{'; '.join(metaphors)}'. "
            f"Incorporate synesthetic impressions: '{'; '.join(synesthesia)}'. "
            f"Reflect the philosophical insights: '{'; '.join(ctul_projections)}'. "
            f"Create a response that is cohesive, resonant, and aligned with the given tone."
        )

        return {
            "prompt": prompt,
            "subject": primary_glyph.glyph_id,
            "emotion": primary_glyph.meta_context.get('emotion', 'neutral') if primary_glyph.meta_context else 'neutral'
        }

if __name__ == "__main__":
    from felt_dto_v3 import FeltDTO
    import numpy as np
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance", "metaphor": "a fleeting shadow"},
        archetypes=["liminal", "desire"],
        vector_embedding=np.random.rand(384).astype(np.float32),
        staunen_markers=[50, 50, 50, 50],
        synesthesia={"light": "ashen peachglow", "sound": "brittle warmth"}
    )
    shaper = PromptShaper()
    prompt_data = shaper.shape_prompt([glyph], task="Write a poetic reflection", tone="poetic")
    print(prompt_data["prompt"])