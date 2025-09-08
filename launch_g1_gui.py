#!/usr/bin/env python3
"""
G1 Inspire GUI Launcher
Launches the G1 simulation with Isaac Sim GUI enabled
"""

import subprocess
import sys
import os

def launch_gui_simulation():
    """Launch G1 Inspire simulation with GUI."""
    
    print("🎮 G1 Inspire GUI Launcher")
    print("=" * 40)
    print("Starting G1 simulation with Isaac Sim GUI...")
    print()
    
    # Set up environment
    unitree_sim_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab"
    
    # Command for GUI mode (no --headless flag)
    cmd = [
        sys.executable,
        os.path.join(unitree_sim_path, "sim_main.py"),
        "--task", "Isaac-PickPlace-RedBlock-G129-Inspire-Joint",
        "--enable_cameras",
        "--action_source", "trajectory"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("🎮 Starting Isaac Sim GUI...")
    print("   The 3D viewport should open shortly")
    print("   You'll see the G1 robot with Inspire hands")
    print("   Red block manipulation environment")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Run with GUI enabled
        process = subprocess.Popen(
            cmd,
            cwd=unitree_sim_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        # Stream output in real-time
        print("📊 Simulation Output:")
        print("-" * 40)
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
                # Look for key indicators
                if "Manager initialized" in line:
                    print("✅ System initialized!")
                elif "Environment created" in line:
                    print("✅ G1 environment loaded!")
                elif "Please left-click" in line:
                    print("🎮 Isaac Sim GUI is ready!")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
        try:
            process.terminate()
            process.wait(timeout=10)
        except:
            process.kill()
        print("✅ Simulation stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    launch_gui_simulation()
