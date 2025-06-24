"""
Epiphany Igniter – Ache-Driven Resonance Spark
----------------------------------------------
Sparks epiphanies by amplifying ache-driven resonance, integrating Zen Entropy Collapse, LLM routing, Maslow Transcendence, and Flow-Passion-Polarization.
"""

from typing import List, Dict
from felt_dto import FeltDTO
from glyph_filter_preset import apply_preset
from dream_queue_writer2 import DreamQueueWriter
from zen_entropy_collapse import ZenEntropyCollapse
from openworld_kframe import OpenWorldGraph
from staunen_amplifier import staunen_amplifier
from glyph_router import GlyphRouter
from maslow_transcendence import MaslowTranscendence
from axiology_vector import AxiologyVector
from flow_passion_polarizer import FlowPassionPolarizer
import numpy as np
import uuid

class EpiphanyIgniter:
    def __init__(self, graph: OpenWorldGraph, openai_key: str):
        self.graph = graph
        self.dream_queue = DreamQueueWriter()
        self.zecp = ZenEntropyCollapse([], graph)
        self.router = GlyphRouter(openai_key=openai_key)
        self.axiology = AxiologyVector()
        self.maslow = MaslowTranscendence(graph, self.axiology)
        self.polarizer = FlowPassionPolarizer(graph, self.axiology)
        self.openai_key = openai_key
        self.staunen_cache = []

    def update_glyph_repository(self, packets: List[FeltDTO]):
        """Updates the glyph repository with new packets, calculating entropy."""
        for packet in packets:
            entropy = self.zecp.calculate_entropy(packet)
            glyph = Glyph(packet.glyph_id, entropy, packet)
            self.zecp.glyph_repository.append(glyph)
            if sum(packet.staunen_markers) / len(packet.staunen_markers) > 50:
                self.staunen_cache.append(packet)

    def ignite_friction(self, packet: FeltDTO, environment: str = "liminal") -> FeltDTO:
        """Ignites epiphany through friction-based resonance (pressure, ache)."""
        if packet.meta_context.get("ache_scalar", 0.0) > 0.7:
            packet = apply_preset(packet, "sunset_gold")
            packet.staunen_markers = [int(s * staunen_amplifier(packet.qualia_map.get("description", ""), environment)) for s in packet.staunen_markers]
            packet.meta_context["epiphany_trigger"] = "friction"
            output = self.router.get_relevant_documents(packet.qualia_map.get("description", ""), packet)
            packet.qualia_map["expert_output"] = output[0]["content"] if output else ""
            packet = self.polarizer.process_glyph(packet)
            packet = self.maslow.process_epiphany(packet)
            self.graph.add_node(packet)
            self.dream_queue.enqueue(packet)
            return packet
        
        for cached in self.staunen_cache:
            if any(a in cached.archetypes for a in packet.archetypes):
                packet.intensity_vector = [v * 1.1 for v in packet.intensity_vector]
                packet.meta_context["epiphany_trigger"] = "mirror_state"
                output = self.router.get_relevant_documents(packet.qualia_map.get("description", ""), packet)
                packet.qualia_map["expert_output"] = output[0]["content"] if output else ""
                packet = self.polarizer.process_glyph(packet)
                packet = self.maslow.process_epiphany(packet)
                self.graph.add_node(packet)
                self.dream_queue.enqueue(packet)
                return packet
        
        return packet

    def ignite_silence(self, count: int = 5, input_text: str = "A hush, a glance") -> FeltDTO:
        """Ignites epiphany through silence-based revelation using ZECP."""
        void_map = self.zecp.generate_void_map(count=count)
        epiphany_dto = self.zecp.trigger_collapse(void_map)
        output = self.router.get_relevant_documents(input_text, epiphany_dto)
        epiphany_dto.qualia_map["expert_output"] = output[0]["content"] if output else ""
        epiphany_dto = self.polarizer.process_glyph(epiphany_dto)
        epiphany_dto = self.maslow.process_epiphany(epiphany_dto)
        if "peak_experience" in epiphany_dto.meta_context.get("transcendence_trigger", ""):
            epiphany_dto.qualia_map["description"] += " — oceanic unity"
        return epiphany_dto

    def process_packet(self, packet: FeltDTO, environment: str = "liminal", input_text: str = "A hush, a glance") -> FeltDTO:
        """Processes a packet, choosing friction or silence-based ignition."""
        self.update_glyph_repository([packet])
        if packet.meta_context.get("ache_scalar", 0.0) > 0.5:
            return self.ignite_friction(packet, environment)
        return self.ignite_silence(count=3, input_text=input_text)

if __name__ == "__main__":
    graph = OpenWorldGraph(graph_id="epiphany_test")
    igniter = EpiphanyIgniter(graph, openai_key="sk-...")
    packet = FeltDTO(
        glyph_id="flow_moment",
        intensity_vector=[0.5, 0.5, 0.5, 0.5],
        meta_context={"emotion": "focus", "ache_scalar": 0.8},
        qualia_map={"sensation": "steady breath"},
        archetypes=["curiosity", "clarity", "unity"],
        staunen_markers=[80, 80, 80, 80],
        vector_embedding=np.random.rand(128).astype(np.float32)
    )
    epiphany = igniter.process_packet(packet, environment="twilight", input_text="A hush, a glance")
    print(f"Epiphany DTO: {epiphany.to_dict()}")