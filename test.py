from felt_dto_v5 import FeltDTO
from staunen_amplifier_v2 import StaunenAmplifier
from glyph_filter_preset import GlyphFilterPreset
from felt_scene import FeltScene, Keyframe
from openworld_kframe import OpenWorldGraph
import numpy as np

glyph = FeltDTO(
    glyph_id="hush_touch",
    intensity_vector=[0.7, 0.6, 0.8, 0.4],
    meta_context={"emotion": "ache", "source": "user", "priority": "high"},
    qualia_map={"description": "A hush, a glance", "metaphor": "a fleeting shadow"},
    archetypes=["liminal", "desire"],
    staunen_markers=[50, 50, 50, 50]
)
amplifier = StaunenAmplifier()
filter = GlyphFilterPreset()
scene = FeltScene()
graph = OpenWorldGraph()

amplified_glyph = amplifier.amplify(glyph)
filtered_glyph = filter.apply_preset(amplified_glyph, "velvia")
scene.keyframes.append(Keyframe(0.0, filtered_glyph))
graph.add_node(filtered_glyph)

print(f"Amplified markers: {amplified_glyph.staunen_markers}")
print(f"Filtered synesthesia: {filtered_glyph.synesthesia}")
print(scene.render())
print(graph.render())