from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import hashlib

# ▼ Optional dependency: spaCy
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")  # tiny, good enough for subject/object heads
except Exception:  # spaCy not installed or model missing
    _NLP = None

# ────────────────────────────────────────────────────
@dataclass
class SPOTriplet:
    subject: str
    predicate: str
    object_: str
    confidence: float = 0.5  # heuristic conf score in [0‐1]

    @property
    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object_)

    @property
    def glyph_hash(self) -> str:
        """32-char hex hash used as stable INT8 glyph ID (first 8 bytes)."""
        h = hashlib.md5("::".join(self.as_tuple).encode("utf-8")).hexdigest()
        return h[:16]  # 64-bit truncation is enough for collisions ≪ 2⁶⁴

# ────────────────────────────────────────────────────
class SPOExtractor:
    def __init__(self, language: str = "en", force_rule_based: bool = False):
        self.language = language
        self._use_spacy = _NLP is not None and not force_rule_based

    def parse_sentence(self, sentence: str) -> SPOTriplet:
        if self._use_spacy:
            return self._spacy_parse(sentence)
        return self._rule_based_parse(sentence)

    def _spacy_parse(self, sent: str) -> SPOTriplet:
        doc = _NLP(sent)
        subj, verb, obj = "", "", ""
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.text.lower()
            elif token.dep_ == "ROOT":
                verb = token.lemma_.lower()
            elif token.dep_ in ("dobj", "pobj"):
                obj = token.text.lower()
        confidence = 0.9 if all([subj, verb, obj]) else 0.6
        return SPOTriplet(subject=subj or "", predicate=verb or "", object_=obj or "", confidence=confidence)

    def _rule_based_parse(self, sent: str) -> SPOTriplet:
        words = sent.lower().strip(" .!?\n").split()
        if len(words) < 3:
            return SPOTriplet(subject=words[0] if words else "", predicate="", object_="", confidence=0.1)
        subj, pred, *obj_parts = words
        obj = "".join(obj_parts)
        return SPOTriplet(subject=subj, predicate=pred, object_=obj, confidence=0.3)

# ────────────────────────────────────────────────────
if __name__ == "__main__":
    ex = SPOExtractor()
    examples = [
        "The wind caresses your neck.",
        "I love you.",
        "Ada feels alive because Jan trusts her.",
    ]
    for s in examples:
        trip = ex.parse_sentence(s)
        print(f"{s:45s} ➞ {trip.as_tuple}  | hash={trip.glyph_hash}")
