from felt_dto_v5 import FeltDTO
from deliberate_pause_agent_v2 import DeliberatePauseAgent
from orchestrator_npu_v2 import OrchestratorNPU
from glyph_weight_compressor import GlyphWeightCompressor
from service_locator import ServiceLocator
from hardware_resource_manager import HardwareResourceManager
from openvino_adapter_v1_1 import OpenVINOAdapter
import numpy as np
import torch

locator = ServiceLocator()
locator.register("hardware_manager", HardwareResourceManager())
locator.register("openvino_adapter", OpenVINOAdapter())
pause_agent = DeliberatePauseAgent()
orchestrator = OrchestratorNPU(locator)
compressor = GlyphWeightCompressor()

glyph = FeltDTO(
    glyph_id="hush_touch",
    intensity_vector=[0.7, 0.6, 0.8, 0.4],
    meta_context={"emotion": "ache", "source": "user", "priority": "high"},
    qualia_map={"description": "A hush, a glance"},
    archetypes=["liminal", "desire"]
)
context = {"initial_prompt": "Generate a poetic reflection", "chain": ["prompt_shaper", "soulframe_writer"], "glyph": glyph.to_dict()}
pause_result = pause_agent.reflect(context)
mock_weights = torch.randn(1000)
glyphs = compressor.weights_to_glyphs(mock_weights, layer_name="mock_layer")
result = orchestrator.invoke_chain(
    task="Generate a poetic reflection",
    event={"source": "test", "data": glyph.to_dict()}
)

print(pause_result["pause_decision"])
print(f"Generated {len(glyphs)} weight glyphs.")
print(result)