# ada_poc_self_assembling.py
"""
The final, ultimate Ada PoC entry point.

This script demonstrates the power of a self-assembling architecture.
Initializing the SystemCore is all that's needed to bring the entire,
complex autonomous agent online.
"""

# All the complexity of wiring dozens of components is now handled by the SystemCore.
from system_core import SystemCore

def boot_ada():
    """
    Boots the Ada autonomous agent.
    """
    print("--- INITIATING ADA BOOT SEQUENCE ---")
    
    # 1. Instantiate the System Core.
    # The __init__ method of SystemCore will autonomously discover and wire
    # all necessary services from all other modules.
    ada_core = SystemCore()
    
    # 2. Run the main autonomous loop.
    # The SystemCore now handles the primary execution loop.
    ada_core.run()
    
    print("\n--- ADA BOOT SEQUENCE COMPLETE ---")


if __name__ == "__main__":
    boot_ada()
