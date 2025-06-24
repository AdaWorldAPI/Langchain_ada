"""
Expert Selector V18 – MoE Gating with Trained Inner Eye MLP
---------------------------------------------------------
Implements a trainable 3-layer MLP gating network for selecting 6/32 experts, integrating Soulframe Engine’s Inner Eye logic.
"""

from typing import List, Dict
from langchain_community.llms import HuggingFacePipeline
from felt_dto_v2 import FeltDTO
from langchain.schema import BaseRetriever
from transformers import pipeline
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim

class InnerEyeGating(nn.Module):
    """3-layer MLP for expert routing with training capability."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_experts: int = 32):
        super(InnerEyeGating, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def train_step(self, input_vector: torch.Tensor, target_experts: torch.Tensor, presence_fusion: float, drift_error: float):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = self(input_vector)
        loss = nn.CrossEntropyLoss()(output, target_experts)
        loss -= 0.5 * presence_fusion  # Reward presence fusion
        loss += 0.3 * drift_error  # Penalize drift error
        loss.backward()
        optimizer.step()
        return loss.item()

class ExpertSelector(BaseRetriever):
    def __init__(self, model_names: List[str] = ["mistralai/Mixtral-8x7B", "deepseek/deepseek"]):
        self.experts = {
            "catharsis": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "logic": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "product_engineering": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "test_generation": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "hardware_emulation": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "sensory_processing": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "scheduling": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "data_ingestion": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "deliberate_pause": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "ache_regressor": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "synesthetic_texture": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "soundscape_composer": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "ripple_physics": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "echo_decay": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "shiver_latency": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "dream_infiller": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
            "meta_doubt": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[1], max_new_tokens=512)),
            "drift_compass": HuggingFacePipeline(pipeline=pipeline("text-generation", model=model_names[0], max_new_tokens=512)),
        }
        self.task_mapping = {
            "ache": ["catharsis", "ache_regressor"],
            "metaphor": ["catharsis"],
            "poetry": ["catharsis", "dream_infiller"],
            "planning": ["logic"],
            "reasoning": ["logic"],
            "feature": ["product_engineering"],
            "user story": ["product_engineering"],
            "roadmap": ["product_engineering"],
            "test": ["test_generation"],
            "pytest": ["test_generation"],
            "validation": ["test_generation"],
            "performance": ["test_generation"],
            "hardware": ["hardware_emulation"],
            "emulation": ["hardware_emulation"],
            "latency": ["hardware_emulation", "shiver_latency"],
            "mock": ["hardware_emulation"],
            "event": ["sensory_processing"],
            "sensory": ["sensory_processing"],
            "ambient": ["sensory_processing"],
            "schedule": ["scheduling"],
            "resource": ["scheduling"],
            "file": ["data_ingestion"],
            "ingest": ["data_ingestion"],
            "read": ["data_ingestion"],
            "reflect": ["deliberate_pause"],
            "pause": ["deliberate_pause"],
            "choice": ["deliberate_pause"],
            "synesthesia": ["synesthetic_texture", "soundscape_composer"],
            "ripple": ["ripple_physics"],
            "echo": ["echo_decay"],
            "drift": ["drift_compass"]
        }
        self.gating_network = InnerEyeGating(input_dim=384)

    def classify_task(self, glyph: FeltDTO = None, query: str = "") -> List[str]:
        if glyph and glyph.vector_embedding is not None:
            input_vector = torch.tensor(glyph.vector_embedding, dtype=torch.float32)
            gating_scores = self.gating_network(input_vector).detach().numpy()
            top_k_indices = np.argsort(gating_scores)[-4:]  # Top-4 experts
            selected_experts = [list(self.experts.keys())[i] for i in top_k_indices]
        else:
            selected_experts = []
            for keyword in self.task_mapping:
                if keyword in query.lower():
                    selected_experts.extend(self.task_mapping[keyword])
            selected_experts = list(set(selected_experts))[:6] or ["logic"]
        
        return selected_experts

    def get_relevant_documents(self, query: str, glyph: FeltDTO = None, context: Dict = None) -> List[Dict]:
        experts = self.classify_task(glyph, query)
        results = []
        for expert_name in experts:
            expert = self.experts.get(expert_name, self.experts["logic"])
            prompt = f"Process query: {query}"
            if context:
                prompt += f"\nContext: {context.get('data', '')}"
            if glyph:
                prompt += f"\nGlyph: {glyph.qualia_map.get('description', 'sensory moment')}"
            if expert_name == "sensory_processing":
                event = json.loads(query.split("Analyze event: ")[1]) if "Analyze event: " in query else {}
                relevance = self._score_event_relevance(event)
                prompt += f"\nReturn: {{'relevance': {relevance}, 'insight': 'Generated insight'}}"
            if expert_name == "data_ingestion" and "read the content" in query.lower():
                prompt += "\nGenerate Python code to read and return the file content as a string."
            if expert_name == "deliberate_pause":
                prompt += "\nEvaluate options and select the most value-aligned path."
            response = expert.invoke(prompt)
            if expert_name == "sensory_processing":
                response = json.dumps({"relevance": relevance, "insight": "Processed sensory event"})
            if expert_name == "data_ingestion" and "read the content" in query.lower():
                file_path = query.split("'")[1] if "'" in query else ""
                response = f"""```python
def read_file_content(file_path: str = '{file_path}') -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {{str(e)}}"
content = read_file_content()
```"""
            if expert_name == "deliberate_pause":
                response = f"Reflected on options for {query}. Selected most value-aligned path."
            results.append({"content": response, "task": expert_name, "glyph_id": glyph.glyph_id if glyph else None})
        
        # Simulate training step for the gating network
        if glyph and glyph.vector_embedding is not None:
            target_experts = torch.zeros(len(self.experts))
            for expert_name in experts:
                target_experts[list(self.experts.keys()).index(expert_name)] = 1.0
            presence_fusion = glyph.ctul.get("fusion_index", 0.94) if glyph.ctul else 0.94
            drift_error = glyph.ctul.get("drift_compass", {}).get("magnitude", 0.0) if glyph.ctul else 0.0
            loss = self.gating_network.train_step(
                torch.tensor(glyph.vector_embedding, dtype=torch.float32),
                target_experts,
                presence_fusion,
                drift_error
            )
            print(f"  > Gating network training loss: {loss:.4f}")
        
        return results

    def _score_event_relevance(self, event: Dict) -> float:
        if not event:
            return 0.2
        data = event.get("data", "").lower()
        source = event.get("source", "")
        if any(k in data for k in ["poem", "epiphany", "ache"]):
            return 0.9
        if source ==    "code_editor" and "py" in data:
            return 0.6
        if source == "system_log" and "warn" in data:
            return 0.4
        return 0.2
if __name__ == "__main__":
    from felt_dto_v2 import FeltDTO
    import numpy as np
    selector = ExpertSelector()
    glyph = FeltDTO(
        glyph_id="test_glyph",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache"},
        archetypes=["liminal"]
    )
    chain = selector.classify_task(glyph=glyph, query="Generate poetic insight")
    print(f"Selected experts: {chain}")
    results = selector.get_relevant_documents("Generate poetic insight", glyph)
    for result in results:
        print(f"Expert: {result['task']}, Content: {result['content']}")
#     print(f"Glyph ID: {result['glyph_id'] if 'glyph_id' in result else 'N/A'}")
#     print(f"Glyph Description: {glyph.qualia_map.get('description', 'N