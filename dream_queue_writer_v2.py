"""
Dream Queue Writer V2 – Robust Glyph Persistence
-----------------------------------------------
Persists FeltDTO glyphs to Redis with retry logic for connection failures.
"""

from typing import List, Dict, Any
from felt_dto_v5 import FeltDTO
import redis
import json
import numpy as np
from service_locator import ServiceLocator
from retrying import retry

class DreamQueueWriter:
    """
    Persists FeltDTO glyphs to Redis with robust connection handling.
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
        print("✅ DreamQueueWriter V2 initialized with Redis connection.")

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def _redis_set(self, key: str, value: str) -> None:
        """Sets a key-value pair in Redis with retry logic."""
        self.redis_client.set(key, value)

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def _redis_lpush(self, key: str, value: str) -> None:
        """Pushes a value to a Redis list with retry logic."""
        self.redis_client.lpush(key, value)

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def _redis_lpop(self, key: str) -> Any:
        """Pops a value from a Redis list with retry logic."""
        return self.redis_client.lpop(key)

    def process_glyph(self, glyph: FeltDTO, reason: str = "default", output: str = "") -> None:
        """
        Processes a FeltDTO glyph, either persisting immediately or enqueuing.

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

        try:
            if glyph.meta_context.get("priority") == "high" or resonance_score > 2.0:
                self._redis_set(f"glyph:{glyph.glyph_id}", json.dumps(glyph_data))
                self.processed_immediately.append(glyph.glyph_id)
                print(f"INFO: Persisted high-priority glyph: {glyph.glyph_id}")
            else:
                self._redis_lpush("dream_queue", json.dumps(glyph_data))
                print(f"INFO: Enqueued low-resonance glyph: {glyph.glyph_id}")
        except redis.RedisError as e:
            print(f"⚠️ Failed to persist glyph {glyph.glyph_id}: {str(e)}")

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
        try:
            self._redis_lpush("dream_queue", json.dumps(glyph_data))
            print(f"INFO: Enqueued glyph: {glyph.glyph_id}")
        except redis.RedisError as e:
            print(f"⚠️ Failed to enqueue glyph {glyph.glyph_id}: {str(e)}")

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
            try:
                raw_data = self._redis_lpop("dream_queue")
                if not raw_data:
                    break
                glyph_data = json.loads(raw_data)
                glyph_id = glyph_data["glyph"]["glyph_id"]
                self._redis_set(f"glyph:{glyph_id}", json.dumps(glyph_data))
                count += 1
                print(f"INFO: Processed batched glyph: {glyph_id}")
            except redis.RedisError as e:
                print(f"⚠️ Failed to process batch: {str(e)}")
                break
        return count