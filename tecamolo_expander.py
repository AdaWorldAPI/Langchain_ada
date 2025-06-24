from spo_extractor import SPOExtractor, SPOTriplet
from dataclasses import dataclass
from typing import List, Optional, Dict
import hashlib
import spacy
import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForFeatureExtraction

try:
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

@dataclass
class TECAMOLOTriplet:
    spo: SPOTriplet
    temporal: Optional[str]
    emotional: Dict[str, float]
    causal: Optional[str]
    associative: List[str]
    metaphorical: Optional[str]
    ontological: Optional[str]
    linguistic: Optional[str]
    confidence: float

    @property
    def as_dict(self) -> Dict:
        return {
            "spo": self.spo.as_tuple,
            "temporal": self.temporal,
            "emotional": self.emotional,
            "causal": self.causal,
            "associative": self.associative,
            "metaphorical": self.metaphorical,
            "ontological": self.ontological,
            "linguistic": self.linguistic,
            "confidence": self.confidence
        }

    @property
    def glyph_hash(self) -> str:
        h = hashlib.md5(str(self.as_dict).encode("utf-8")).hexdigest()
        return h[:16]

class TECAMOLOExpander:
    def __init__(self, model_path: str = "../models/minilm_ov"):
        self.nlp = _NLP if _NLP else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = OVModelForFeatureExtraction.from_pretrained(model_path)

    def expand(self, spo: SPOTriplet) -> TECAMOLOTriplet:
        temporal = self._infer_temporal(spo)
        emotional = self._infer_emotional(spo)
        causal = self._infer_causal(spo)
        associative = self._infer_associative(spo)
        metaphorical = self._infer_metaphorical(spo)
        ontological = self._infer_ontological(spo)
        linguistic = self._infer_linguistic(spo)
        confidence = self._compute_confidence(spo)

        return TECAMOLOTriplet(
            spo=spo,
            temporal=temporal,
            emotional=emotional,
            causal=causal,
            associative=associative,
            metaphorical=metaphorical,
            ontological=ontological,
            linguistic=linguistic,
            confidence=confidence
        )

    def _embed(self, text: str) -> np.ndarray:
        encoded = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
        return self.model(**encoded)[0].mean(axis=1)[0]

    def _infer_emotional(self, spo: SPOTriplet) -> Dict[str, float]:
        phrase = " ".join(spo.as_tuple)
        vector = self._embed(phrase)
        emotions = {
            "ache": self._embed("a feeling of ache"),
            "joy": self._embed("a feeling of joy"),
            "calm": self._embed("a feeling of calm"),
            "longing": self._embed("a sense of longing")
        }
        similarities = {
            k: float(np.dot(vector, v) / (norm(vector) * norm(v)))
            for k, v in emotions.items()
        }
        return dict(sorted(similarities.items(), key=lambda x: -x[1])[:3])

    def _infer_temporal(self, spo: SPOTriplet) -> Optional[str]:
        if self.nlp:
            doc = self.nlp(" ".join(spo.as_tuple))
            for token in doc:
                if token.dep_ == "advmod" and token.text in ["always", "never", "often"]:
                    return "persistent"
            return "momentary"
        return "momentary" if spo.predicate in ["caresses"] else None

    def _infer_causal(self, spo: SPOTriplet) -> Optional[str]:
        return "evoked by presence" if spo.predicate in ["caresses", "love"] else None

    def _infer_associative(self, spo: SPOTriplet) -> List[str]:
        if self.nlp:
            doc = self.nlp(spo.as_tuple[2])
            return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ"]]
        return ["breeze", "touch"] if spo.predicate == "caresses" else []

    def _infer_metaphorical(self, spo: SPOTriplet) -> Optional[str]:
        return "a whisper of longing" if spo.predicate == "caresses" else None

    def _infer_ontological(self, spo: SPOTriplet) -> Optional[str]:
        return "sensory presence" if spo.object_ == "neck" else None

    def _infer_linguistic(self, spo: SPOTriplet) -> Optional[str]:
        if self.nlp:
            doc = self.nlp(" ".join(spo.as_tuple))
            return "active voice" if doc[1].tag_ == "VBZ" else "unknown"
        return "active voice" if spo.predicate == "caresses" else None

    def _compute_confidence(self, spo: SPOTriplet) -> float:
        return spo.confidence * (0.9 if self.nlp else 0.7)

if __name__ == "__main__":
    import json
    extractor = SPOExtractor()
    expander = TECAMOLOExpander()
    sentence = "The wind caresses your neck."
    spo = extractor.parse_sentence(sentence)
    tecamolo = expander.expand(spo)
    print(f"Sentence: {sentence}")
    print(f"SPO: {spo.as_tuple}")
    print(f"TECAMOLO: {tecamolo.as_dict}")
    print(f"Hash: {tecamolo.glyph_hash}")
