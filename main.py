# scripts/main.py
"""
Main application to run Ada's overnight felt embedding and retrieval pipeline.

This script:
1. Loads sample FeltDTO-like data.
2. Loads the pre-converted OpenVINO embedding model.
3. Generates vector embeddings for the sample data.
4. Builds a FAISS index for efficient similarity search.
5. Performs a sample query to find the most relevant "felt" moment.
"""
import numpy as np
import faiss
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForFeatureExtraction
import os
from tqdm import tqdm

# --- FeltDTO Definition ---
# Copied here from your Canvas for a self-contained, runnable script.
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
import time

@dataclass
class FeltDTO:
    glyph_id: str
    vector_embedding: np.ndarray = None # Will be populated later
    meta_context: Dict[str, Any] = field(default_factory=dict)
    qualia_map: Dict[str, Any] = field(default_factory=dict)
    # Adding a text field to hold the source content for embedding
    source_text: str = ""

# --- Main Pipeline ---

# Configuration
OV_MODEL_PATH = os.path.join("..", "models", "minilm_l6_ov")
DIMENSIONS = 384  # Dimensions of the MiniLM-L6-H384 model

def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on the token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def run_pipeline():
    """Executes the full embedding and retrieval pipeline."""
    print("--- Starting Ada's Introspection Pipeline ---")

    # 1. Load sample felt inputs
    print("Step 1: Loading sample 'felt' moments...")
    felt_inputs = [
        FeltDTO(glyph_id="rilke_der_panther_v1", source_text="His gaze has from the passing of the bars grown so weary that it holds nothing more."),
        FeltDTO(glyph_id="hush_of_first_snow", source_text="A quiet moment of peace during a cold winter's night."),
        FeltDTO(glyph_id="ache_of_distance", source_text="A profound feeling of longing and separation from a loved one."),
        FeltDTO(glyph_id="storm_aftermath_stillness", source_text="The silence after the storm was a loud reminder of the fragility of peace."),
    ]
    texts_to_embed = [f.source_text for f in felt_inputs]
    print(f"  > Loaded {len(felt_inputs)} moments.")

    # 2. Load the optimized OpenVINO model and tokenizer
    print(f"Step 2: Loading OpenVINO model from '{OV_MODEL_PATH}'...")
    if not os.path.exists(OV_MODEL_PATH):
        print(f"  > ERROR: Model not found. Please run 'convert_model.py' first.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(OV_MODEL_PATH)
    ov_model = OVModelForFeatureExtraction.from_pretrained(OV_MODEL_PATH, device="CPU") # Explicitly use CPU
    print("  > Model loaded successfully.")

    # 3. Tokenize and encode the inputs to generate embeddings
    print("Step 3: Generating vector embeddings...")
    encoded_input = tokenizer(texts_to_embed, padding=True, truncation=True, return_tensors="np")
    model_output = ov_model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings for better cosine similarity search
    faiss.normalize_L2(embeddings)

    # Assign embeddings back to the DTOs
    for i, felt in enumerate(felt_inputs):
        felt.vector_embedding = embeddings[i]
    print(f"  > Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")

    # 4. Build and populate the FAISS index
    print("Step 4: Building FAISS index...")
    index = faiss.IndexFlatIP(DIMENSIONS) # Using Inner Product for normalized vectors
    index.add(embeddings)
    print(f"  > FAISS index created with {index.ntotal} vectors.")

    # 5. Query the index with a test prompt
    print("\n--- Running Retrieval Test ---")
    query_text = "a memory of loss and feeling alone"
    print(f"Query: '{query_text}'")

    # Generate embedding for the query
    encoded_query = tokenizer([query_text], padding=True, truncation=True, return_tensors="np")
    query_output = ov_model(**encoded_query)
    query_embedding = mean_pooling(query_output, encoded_query["attention_mask"])
    faiss.normalize_L2(query_embedding)

    # Search the index
    k = 1 # Find the single best match
    distances, indices = index.search(query_embedding, k)

    # Display results
    print("\n--- Search Results ---")
    for i in range(k):
        result_index = indices[0][i]
        retrieved_felt = felt_inputs[result_index]
        similarity = distances[0][i]
        print(f"Top Match: '{retrieved_felt.glyph_id}' (Similarity: {similarity:.4f})")
        print(f"  Source Text: '{retrieved_felt.source_text}'")

if __name__ == "__main__":
    run_pipeline()
