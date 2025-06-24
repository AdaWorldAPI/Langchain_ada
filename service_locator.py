# service_locator.py
"""
Defines the ServiceLocator, a central registry for all system services.
This enables a decoupled architecture where components can request their
dependencies without needing to know how they are created or managed.
"""
from typing import Dict, Any

class ServiceLocator:
    """A central registry for system services."""
    def __init__(self):
        self._services: Dict[str, Any] = {}
        print("✅ ServiceLocator initialized.")

    def register(self, name: str, service_instance: Any):
        """Registers a service instance with a given name."""
        print(f"  > Registering service: '{name}'")
        self._services[name] = service_instance

    def get(self, name: str) -> Any:
        """Retrieves a service instance by name."""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not found in ServiceLocator.")
        return self._services[name]

# Create a global instance to be used throughout the system
locator = ServiceLocator()