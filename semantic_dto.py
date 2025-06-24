"""
Stage 1c – Semantic Aggregation ▸ SemanticDTO
--------------------------------------------
Aggregates TECAMOLO contexts into structured semantic representations, optimized for FAISS and INT8.
"""

from dataclasses import dataclass
from typing import List, Dict
import hashlib
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from tecamolo_expander import TECAMOLOTriplet

@dataclass
class SemanticDTO:
    tecamolo: TECAMOLOTriplet
    semantic_context: Dict[str, str]
    entities: List[str]
    relations: List[str]
    confidence: float

    @property
    def as_dict(self) -> Dict:
        return {
            "tecamolo": self.tecamolo.as_dict,
            "semantic_context": self.semantic_context,
            "entities": self.entities,
            "relations": self.relations,
            "confidence": self.confidence
        }

    @property
    def glyph_hash(self) -> str:
        h = hashlib.md5(str(self.as_dict).encode("utf-8")).hexdigest()
        return h[:16]

class SemanticAggregator:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def aggregate(self, tecamolo: TECAMOLOTriplet) -> SemanticDTO:
        semantic_context = {
            "intent": "express longing" if tecamolo.emotional.get("ache", 0) > 0.5 else "describe sensation",
            "domain": "emotion" if tecamolo.emotional else "neutral"
        }
        entities = [tecamolo.spo.subject, tecamolo.spo.object_] + tecamolo.associative
        relations = [tecamolo.spo.predicate] + ([tecamolo.causal] if tecamolo.causal else [])
        confidence = tecamolo.confidence * 0.95

        return SemanticDTO(
            tecamolo=tecamolo,
            semantic_context=semantic_context,
            entities=entities,
            relations=relations,
            confidence=confidence
        )

if __name__ == "__main__":
    from spo_extractor import SPOExtractor
    from tecamolo_expander import TECAMOLOExpander

    extractor = SPOExtractor()
    expander = TECAMOLOExpander()
    aggregator = SemanticAggregator()

    sentence = "The wind caresses your neck."
    spo = extractor.parse_sentence(sentence)
    tecamolo = expander.expand(spo)
    semantic = aggregator.aggregate(tecamolo)

    print(f"Sentence: {sentence}")
    print(f"SPO: {spo.as_tuple}")
    print(f"TECAMOLO: {tecamolo.as_dict}")
    print(f"SemanticDTO: {semantic.as_dict}")
    print(f"Hash: {semantic.glyph_hash}")
