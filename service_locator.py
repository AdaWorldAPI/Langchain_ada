"""
Service Locator – Dependency Injection
-------------------------------------
Provides a service locator for dependency injection in Ada’s architecture.
"""

from typing import Any, Dict

class ServiceLocator:
    """
    Singleton service locator for managing dependencies.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceLocator, cls).__new__(cls)
            cls._instance.services: Dict[str, Any] = {}
        return cls._instance

    def register(self, service_name: str, service_instance: Any) -> None:
        """
        Registers a service instance.

        Args:
            service_name: Name of the service.
            service_instance: Instance of the service.
        """
        self.services[service_name] = service_instance
        print(f"✅ Registered service: {service_name}")

    def get(self, service_name: str) -> Any:
        """
        Retrieves a service instance by name.

        Args:
            service_name: Name of the service to retrieve.

        Returns:
            Service instance or None if not found.
        """
        return self.services.get(service_name)

if __name__ == "__main__":
    locator = ServiceLocator()
    locator.register("test_service", object())
    print(locator.get("test_service"))