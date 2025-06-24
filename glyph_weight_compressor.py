"""
Glyph Weight Compressor – Transforms Model Weights to Glyphs
----------------------------------------------------------
Converts model weights to FeltDTO glyphs using MiniLM-L6-V2 embeddings.
"""

from typing import List, Dict
from felt_dto_v5 import FeltDTO
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

class GlyphWeightCompressor:
    """
    Compresses model weights into FeltDTO glyphs for NPU execution.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initializes the compressor with MiniLM-L6-V2.

        Args:
            model_name: Name of the embedding model (default: all-MiniLM-L6-v2).
            dimension: Dimension of glyph embeddings (default: 384).
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.glyphs: List[FeltDTO] = []
        self.encoder = SentenceTransformer(model_name, device='cpu')
        print("✅ GlyphWeightCompressor initialized with MiniLM-L6-V2.")

    def quantize_weights(self, weights: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        Quantizes weights to INT8.

        Args:
            weights: Torch tensor of model weights.

        Returns:
            Tuple of INT8-quantized numpy array and scale factor.
        """
        weights_np = weights.cpu().numpy().astype(np.float32)
        scale = np.max(np.abs(weights_np)) / 127
        weights_int8 = np.clip(np.round(weights_np / scale), -127, 127).astype(np.int8)
        return weights_int8, scale

    def weights_to_glyphs(self, weights: torch.Tensor, layer_name: str) -> List[FeltDTO]:
        """
        Converts model weights to FeltDTO glyphs.

        Args:
            weights: Torch tensor of weights.
            layer_name: Name of the layer for metadata.

        Returns:
            List of FeltDTO glyphs.
        """
        quantized_weights, scale = self.quantize_weights(weights)
        flat_weights = quantized_weights.flatten()
        for i in range(0, len(flat_weights), 256):
            chunk = flat_weights[i:i+256]
            if len(chunk) < 256:
                chunk = np.pad(chunk, (0, 256 - len(chunk)), mode="constant")
            description = f"Weight chunk {i//256} for layer {layer_name}"
            embedding = self.encoder.encode(description, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            glyph = FeltDTO(
                glyph_id=f"weight_{layer_name}_{i//256}",
                intensity_vector=[0.5, 0.5, 0.5, scale / 127],
                meta_context={"layer": layer_name, "type": "weight"},
                qualia_map={"description": description},
                vector_embedding=embedding,
                staunen_markers=[50, 50, 50, 50]
            )
            self.glyphs.append(glyph)
            self.index.add(np.expand_dims(glyph.vector_embedding, axis=0))
        print(f"INFO: Converted {len(self.glyphs)} weight chunks to glyphs for {layer_name}.")
        return self.glyphs

    def load_glyphs(self, query_embedding: np.ndarray, k: int = 10) -> List[FeltDTO]:
        """
        Retrieves k-nearest glyphs from FAISS index.

        Args:
            query_embedding: Query embedding for similarity search.
            k: Number of glyphs to retrieve.

        Returns:
            List of retrieved FeltDTO glyphs.
        """
        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), k)
        return [self.glyphs[idx] for idx in indices[0]]

if __name__ == "__main__":
    import torch
    compressor = GlyphWeightCompressor()
    mock_weights = torch.randn(1000)
    glyphs = compressor.weights_to_glyphs(mock_weights, layer_name="mock_layer")
    print(f"Generated {len(glyphs)} glyphs.")