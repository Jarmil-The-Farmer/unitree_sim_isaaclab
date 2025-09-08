#!/usr/bin/env python3
"""
G1 Inspire Simulation with Full Isaac Sim GUI
Direct launch with visualization enabled
"""

import subprocess
import sys
import os
import time

def main():
    """Launch G1 Inspire simulation with full Isaac Sim GUI."""
    
    print("🎮 G1 Inspire Simulation with Isaac Sim GUI")
    print("=" * 50)
    print("This will open Isaac Sim with full 3D visualization")
    print()
    
    # Change to unitree sim directory
    unitree_dir = "/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab"
    os.chdir(unitree_dir)
    
    # Command for GUI mode (remove --headless)
    cmd = [
        sys.executable,
        "sim_main.py",
        "--task", "Isaac-PickPlace-RedBlock-G129-Inspire-Joint",
        # NO --headless flag = GUI enabled
        "--enable_cameras",
        "--action_source", "trajectory",  # Safe trajectory mode
        "--step_hz", "60"
    ]
    
    print(f"🚀 Starting Isaac Sim GUI...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {unitree_dir}")
    print()
    print("📺 Isaac Sim should open with:")
    print("   ✅ Full 3D viewport")
    print("   ✅ G1 robot with Inspire hands")
    print("   ✅ PickPlace environment with red block")
    print("   ✅ Camera views")
    print("   ✅ Real-time physics simulation")
    print()
    print("🎮 In Isaac Sim you can:")
    print("   • Rotate/zoom the 3D view")
    print("   • Watch the G1 robot in action")
    print("   • See the Inspire hands manipulating objects")
    print("   • Monitor camera feeds")
    print("   • Use Play/Pause controls")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Run the simulation with GUI
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("🎬 Isaac Sim GUI starting...")
        
        # Monitor the output
        line_count = 0
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                    
                line = line.strip()
                if line:
                    line_count += 1
                    
                    # Show important startup messages
                    if any(keyword in line.lower() for keyword in [
                        'loading', 'environment', 'robot', 'camera', 'dds', 
                        'simulation', 'isaac', 'g1', 'inspire'
                    ]):
                        print(f"📋 {line}")
                    
                    # Show warnings and errors
                    elif any(keyword in line.lower() for keyword in ['warning', 'error']):
                        if 'material' not in line.lower():  # Skip material warnings
                            print(f"⚠️  {line}")
                    
                    # Detect when GUI is ready
                    if "environment created" in line.lower() or "simulation ready" in line.lower():
                        print("🎉 Isaac Sim GUI is ready!")
                        print("   You should now see the G1 robot in the 3D viewport")
                
                # Check if process ended
                if process.poll() is not None:
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping simulation...")
                process.terminate()
                break
            except Exception as e:
                print(f"⚠️  Error reading output: {e}")
                break
        
        # Wait for process to finish
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print("✅ Simulation ended")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error starting simulation: {e}")
    
    print("\n🎯 To restart with visualization:")
    print("   python launch_gui_simulation.py")

if __name__ == "__main__":
    main()
