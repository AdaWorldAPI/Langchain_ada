# qualia_input_stream.py
"""
Defines the QualiaInputStream, a component that simulates a continuous stream
of sensory data from the system's environment.
"""
import time
import random
from typing import Generator, Dict, Any

class QualiaInputStream:
    """Simulates a stream of ambient sensory events."""
    def __init__(self):
        self._event_sources = [
            ("system_log", "INFO: Scheduled disk cleanup complete."),
            ("file_system", "User saved a new file: 'notes_on_ache.txt'."),
            ("system_log", "DEBUG: TCP connection timed out."),
            ("code_editor", "User is editing `soulframe_writer.py`."),
            ("system_log", "WARN: Memory usage at 85%."),
            ("file_system", "User saved a new file: 'epiphany_poem.md'.")
        ]
        print("✅ QualiaInputStream initialized.")

    def listen(self) -> Generator[Dict[str, Any], None, None]:
        """A generator that yields a new sensory event every few seconds."""
        while True:
            # In a real system, this would be an async listener. Here, we simulate.
            time.sleep(random.uniform(2, 5)) 
            event_type, event_data = random.choice(self._event_sources)
            yield {
                "timestamp": time.time(),
                "source": event_type,
                "data": event_data
            }

