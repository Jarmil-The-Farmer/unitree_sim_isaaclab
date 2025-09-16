#!/usr/bin/env python3
"""
Direct G1 GUI Command
Run this for immediate GUI visualization
"""

import subprocess
import sys
import os

# Simple direct command
cmd = [
    sys.executable,
    "/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab/sim_main.py",
    "--task", "Isaac-PickPlace-RedBlock-G129-Inspire-Joint", 
    "--enable_cameras"
]

print("🚀 Starting G1 Inspire with Isaac Sim GUI...")
print(f"Command: {' '.join(cmd)}")

os.chdir("/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab")
subprocess.run(cmd)
