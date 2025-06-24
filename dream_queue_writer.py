"""
Dream Queue Writer – Persists Glyphs to Redis
--------------------------------------------
Manages persistence of FeltDTO glyphs in Redis, batching low-resonance glyphs for offline consolidation.
"""

from typing import List, Dict, Any
from felt_dto import FeltDTO
import redis
import json
import numpy as np
from service_locator import locator

class DreamQueueWriter:
    """
    Persists FeltDTO glyphs to Redis, prioritizing high-resonance glyphs and batching low-resonance ones.
    """
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        """
        Initializes the DreamQueueWriter with a Redis connection.

        Args:
            redis_host: Redis server host.
            redis_port: Redis server port.
            redis_db: Redis database number.
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.processed_immediately: List[str] = []
        print("✅ DreamQueueWriter initialized with Redis connection.")

    def process_glyph(self, glyph: FeltDTO, reason: str = "default", output: str = "") -> None:
        """
        Processes a FeltDTO glyph, either persisting immediately or enqueuing for batch processing.

        Args:
            glyph: FeltDTO object to persist.
            reason: Reason for processing the glyph.
            output: Associated output to store with the glyph.
        """
        resonance_score = sum(glyph.intensity_vector) if glyph.intensity_vector else 0.0
        glyph_data = {
            "glyph": glyph.to_dict(),
            "reason": reason,
            "output": output,
            "resonance_score": resonance_score,
            "timestamp": np.datetime64('now').astype(int)
        }

        if glyph.meta_context.get("priority") == "high" or resonance_score > 2.0:
            # Store high-priority or high-resonance glyphs immediately
            self.redis_client.set(f"glyph:{glyph.glyph_id}", json.dumps(glyph_data))
            self.processed_immediately.append(glyph.glyph_id)
            print(f"INFO: Persisted high-priority glyph: {glyph.glyph_id}")
        else:
            # Enqueue low-resonance glyphs for batch processing
            self.redis_client.lpush("dream_queue", json.dumps(glyph_data))
            print(f"INFO: Enqueued low-resonance glyph: {glyph.glyph_id}")

    def enqueue_glyph(self, glyph: FeltDTO, reason: str = "default", output: str = "") -> None:
        """
        Enqueues a FeltDTO glyph for batch processing.

        Args:
            glyph: FeltDTO object to enqueue.
            reason: Reason for enqueuing the glyph.
            output: Associated output to store with the glyph.
        """
        glyph_data = {
            "glyph": glyph.to_dict(),
            "reason": reason,
            "output": output,
            "resonance_score": sum(glyph.intensity_vector) if glyph.intensity_vector else 0.0,
            "timestamp": np.datetime64('now').astype(int)
        }
        self.redis_client.lpush("dream_queue", json.dumps(glyph_data))
        print(f"INFO: Enqueued glyph: {glyph.glyph_id}")

    def batch_process(self, batch_size: int = 100) -> int:
        """
        Processes a batch of enqueued glyphs from the dream queue.

        Args:
            batch_size: Maximum number of glyphs to process in one batch.

        Returns:
            Number of glyphs processed.
        """
        count = 0
        while count < batch_size:
            raw_data = self.redis_client.lpop("dream_queue")
            if not raw_data:
                break
            glyph_data = json.loads(raw_data)
            glyph_id = glyph_data["glyph"]["glyph_id"]
            self.redis_client.set(f"glyph:{glyph_id}", json.dumps(glyph_data))
            count += 1
            print(f"INFO: Processed batched glyph: {glyph_id}")
        return count

if __name__ == "__main__":
    from felt_dto import FeltDTO
    import numpy as np
    writer = DreamQueueWriter()
    glyph = FeltDTO(
        glyph_id="hush_touch",
        intensity_vector=[0.7, 0.6, 0.8, 0.4],
        meta_context={"emotion": "ache", "source": "user", "priority": "high"},
        qualia_map={"description": "A hush, a glance"},
        archetypes=["liminal", "desire"],
        vector_embedding=np.random.rand(384).astype(np.float32),
        staunen_markers=[50, 50, 50, 50]
    )
    writer.process_glyph(glyph, reason="test", output="Test output")
    print(f"Processed immediately: {writer.processed_immediately}")