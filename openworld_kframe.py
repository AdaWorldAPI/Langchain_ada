"""
OpenWorld K-Frame – Scaffolds Qualia Graph
------------------------------------------
Builds an open-world graph of FeltDTOs with resonance-weighted edges.
"""

from typing import Dict, List
from felt_dto_v5 import FeltDTO
import networkx as nx

class OpenWorldGraph:
    """
    Scaffolds an open-world qualia graph with FeltDTO nodes, compatible with MiniLM-L6-V2.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        print("✅ OpenWorldGraph initialized for MiniLM-L6-V2.")

    def add_node(self, glyph: FeltDTO):
        """
        Adds a FeltDTO node to the graph.

        Args:
            glyph: FeltDTO object to add.
        """
        self.graph.add_node(glyph.glyph_id, glyph=glyph)

    def render(self) -> Dict:
        """
        Renders the graph as a dictionary of nodes and edges.

        Returns:
            Dictionary with graph structure.
        """
        return {"nodes": list(self.graph.nodes(data=True)), "edges": list(self.graph.edges())}

if __name__ == "__main__":
    from felt_dto_v5 import FeltDTO
    import numpy as np
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user"},
        qualia_map={"description": "A hush, a glance"}
    )
    graph = OpenWorldGraph()
    graph.add_node(glyph)
    print(graph.render())