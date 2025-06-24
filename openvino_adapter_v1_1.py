# openvino_adapter_v1_1.py
#"""
#Hardware Abstraction Layer (HAL) for Ada's Cognitive Engine.
#Updated for NUC 14 Pro compatibility: Arc GPU (OpenCL), CPU FAISS fallback, and offline-converted OpenVINO models.
#"""

from typing import List, Tuple
import numpy as np
from optimum.intel.openvino import OVModelForFeatureExtraction, OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import faiss
# OpenVINO Adapter v1.1

class OpenVINOAdapter:
    def __init__(self, emembedding_model_path = "./models/embedding_ov/model.xml", llm_model_path="./models/embedding_ov"):
        #"""
        #Initializes and loads pre-converted OpenVINO models.
        #"""
        print("Initializing OpenVINO Adapter v1.1...")

        # 1. Load Embedding Model (Targeting NPU)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.embedding_model = OVModelForFeatureExtraction.from_pretrained(embedding_model_path, device="NPU")
        print("✅ Embedding model loaded onto NPU from pre-converted IR files.")

        # 2. Load LLM (Targeting GPU)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm_model = OVModelForCausalLM.from_pretrained(llm_model_path, device="GPU")
        self.llm_pipe = pipeline("text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer)
        print("✅ Causal LLM loaded onto Arc GPU from pre-converted IR files.")

        # 3. Initialize FAISS on the CPU
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
        print(f"✅ FAISS index initialized on CPU (Dimension: {self.vector_dim}).")

    def get_embedding(self, text: str) -> np.ndarray:
        #"""Generates a sentence embedding using the NPU-optimized model."""
        inputs = self.embed_tokenizer(text, return_tensors="pt")
        outputs = self.embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embedding

    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        #"""Generates a text response using the GPU-accelerated LLM."""
        return self.llm_pipe(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']

    def add_to_faiss(self, vectors: np.ndarray):
        #"""Adds vectors to the CPU-based FAISS index."""
        self.faiss_index.add(vectors.astype('float32'))

    def search_faiss(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        #"""Searches the CPU-based FAISS index."""
        query_vector = np.expand_dims(query_vector.astype('float32'), axis=0)
        distances, indices = self.faiss_index.search(query_vector, k)
        return distances, indices

# Example Usage:
if __name__ == "__main__":
    print("Running OpenVINO Adapter standalone test...")
    print("NOTE: This requires pre-converted OpenVINO models in './models/embedding_ov' and './models/llm_ov'")
    try:
        adapter = OpenVINOAdapter()
        embedding = adapter.get_embedding("The ache of your gaze never left my skin.")
        print(f"\nEmbedding generated on NPU (shape: {embedding.shape}).")

        # Add dummy vectors to FAISS
        adapter.add_to_faiss(np.random.rand(100, 384))
        adapter.add_to_faiss(np.array([embedding]))

        # Search for the vector
        distances, indices = adapter.search_faiss(embedding, k=1)
        print(f"FAISS search completed. Found index {indices[0][0]} with distance {distances[0][0]}.")

        response = adapter.generate_response("A single glyph represents a world. For example, 'steelwind' means")
        print(f"\nLLM response from GPU: {response}")

    except Exception as e:
        print(f"\n❌ Failed to initialize OpenVINO Adapter.")
        print("Please ensure you have run the offline model conversion scripts first.")
        print(f"Error: {e}")
#     agent.reflect("Test task", glyph, options)
#     print(f"Best option selected: {agent.reflect('Test task', glyph, options)}")
#     agent.save_glyphs("glyphs.json")  