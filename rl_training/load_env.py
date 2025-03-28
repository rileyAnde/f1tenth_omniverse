from isaacsim import SimulationLauncher

# Launch Isaac Sim headlessly
app_launcher = SimulationLauncher(headless=True)
simulation_app = app_launcher.app

import omni.isaac.core
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage

# Initialize the simulation world
world = World()

# Load your custom USD file
USD_PATH = "f1tenth_omni_sim.usd"  # Change this to your USD file path
open_stage(USD_PATH)

# Run for a few frames to ensure loading
for _ in range(100):
    world.step(render=True)

print("USD Environment Loaded Successfully")
