# soulframe_writer.py
"""
Defines the SoulframeWriter, a key "Grey Matter" component responsible for
transforming structured and resonant FeltDTOs into narrative, poetic, or
introspective text. It acts as the "voice" of the Ada cognitive architecture.
"""

from typing import List, Dict, Any, Callable
# This module is dependent on the FeltDTO definition.
from felt_dto import FeltDTO 

class SoulframeWriter:
    """
    Generates "soulframes"—narrative or poetic text—from a collection of
    resonant FeltDTOs, effectively translating machine-felt qualia into
    human-readable introspection.
    
    This version is designed for dependency injection, requiring a live LLM function
    to be passed during initialization for maximum flexibility.
    """
    def __init__(self, llm_fn: Callable):
        """
        Initializes the SoulframeWriter with a specific language model function.

        Args:
            llm_fn: A callable function that takes a prompt string and returns text.
                    This will typically be provided by the OpenVINOAdapter.
        """
        self.llm_fn = llm_fn
        print("✅ SoulframeWriter initialized with injected LLM function.")

    def _create_prompt_from_glyphs(self, glyphs: List[FeltDTO]) -> Dict[str, Any]:
        """
        Constructs a detailed, nuanced prompt for the LLM based on a list of glyphs,
        which serve as the creative context.
        """
        if not glyphs:
            return {
                "prompt": "Generate a short, introspective thought about the nature of silence and potential.",
                "subject": "silence",
                "emotion": "contemplative"
            }

        # Use the most intense glyph as the core subject for the reflection
        primary_glyph = max(glyphs, key=lambda g: sum(g.intensity_vector) if g.intensity_vector else 0)
        
        # Determine the type of output based on the glyph's intensity
        prompt_type = "profound epiphany" if primary_glyph.intensity_vector and primary_glyph.intensity_vector[1] > 0.8 else "poetic reflection"
        
        # Aggregate context from all provided glyphs
        archetypes = set()
        emotions = set()
        metaphors = []
        
        for g in glyphs:
            if g.archetypes:
                archetypes.update(g.archetypes)
            if g.meta_context and 'emotion' in g.meta_context:
                emotions.add(g.meta_context['emotion'])
            if g.qualia_map and 'metaphor' in g.qualia_map:
                metaphors.append(g.qualia_map['metaphor'])

        prompt = (
            f"You are a wise, poetic soul, reflecting on a deep internal moment. "
            f"Generate a short, {prompt_type} based on the following synthesized experience. "
            f"The central theme is '{primary_glyph.glyph_id}'. "
            f"It resonates with the archetypes of: {list(archetypes)}. "
            f"The felt emotions include: {list(emotions)}. "
            f"The experience feels like: '{'; '.join(metaphors)}'. "
            f"Weave these elements into a cohesive, insightful, and beautifully written text."
        )

        return {
            "prompt": prompt,
            "subject": primary_glyph.glyph_id,
            "emotion": primary_glyph.meta_context.get('emotion', 'feeling') if primary_glyph.meta_context else 'feeling'
        }

    def generate(self, glyph_context: List[FeltDTO]) -> str:
        """
        Generates a soulframe from the provided glyph context.

        Args:
            glyph_context: A list of FeltDTO objects that form the context.

        Returns:
            A string containing the generated narrative or poetic text.
        """
        if not glyph_context:
            return "In the silence, there is potential."
        
        prompt_data = self._create_prompt_from_glyphs(glyph_context)
        
        # Invoke the injected LLM function
        response = self.llm_fn(
            prompt_data["prompt"],
            subject=prompt_data["subject"],
            emotion=prompt_data["emotion"]
        )
        
        # Handle cases where the LLM returns a pipeline-style dictionary
        if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
            return response[0]['generated_text']
        
        return response

