#!/usr/bin/env python3
"""
Isaac Sim GUI Launch Script for G1 Inspire + LeRobot
Optimized for full visualization experience
"""

import os
import sys
import subprocess
import time

def launch_with_visualization():
    """Launch the G1 Inspire simulation with full Isaac Sim visualization."""
    
    print("🎮 Isaac Sim GUI Launch for G1 Inspire + LeRobot")
    print("=" * 60)
    print("This will open Isaac Sim with full GUI and 3D visualization")
    print()
    
    # Ensure we're in the right conda environment
    if "unitree_sim_env" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("⚠️  Please activate the unitree_sim_env conda environment first:")
        print("   conda activate unitree_sim_env")
        return False
    
    # Option 1: Direct visualization with working subprocess approach
    print("🚀 Option 1: Full Isaac Sim GUI with G1 Inspire simulation")
    print("This will show:")
    print("   • Complete Isaac Sim interface")
    print("   • 3D viewport with G1 robot and Inspire hands")
    print("   • PickPlace environment with red block")
    print("   • Real-time robot visualization")
    print("   • Camera feeds and sensor data")
    print()
    
    # Launch the visualization script
    launch_direct_visualization()
    
    return True

def launch_direct_visualization():
    """Launch with direct Isaac Sim visualization."""
    
    print("🎬 Starting Isaac Sim with G1 Inspire visualization...")
    
    # Use the working integration that bypasses library conflicts
    cmd = [
        sys.executable,
        "/home/vaclav/IsaacLab_unitree/IsaacLab/run_g1_inspire_direct.py"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("📺 Isaac Sim GUI should open showing:")
    print("   ✅ G1 robot with Inspire hands (50+ DOF)")
    print("   ✅ Isaac-PickPlace-RedBlock environment")
    print("   ✅ Stereo cameras providing visual feedback")
    print("   ✅ Real-time physics simulation")
    print("   ✅ LeRobot-compatible environment")
    print()
    print("🎮 In the Isaac Sim GUI you'll see:")
    print("   • 3D viewport with the complete scene")
    print("   • G1 robot performing manipulation tasks")
    print("   • Camera windows showing stereo vision")
    print("   • Physics simulation running at 60Hz")
    print()
    print("Press Ctrl+C to stop the simulation")
    print("=" * 60)
    
    try:
        # Run the visualization
        subprocess.run(cmd, check=True)
        print("✅ Visualization completed successfully")
        
    except KeyboardInterrupt:
        print("\\n🛑 Visualization stopped by user")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization failed with error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def launch_alternative_options():
    """Show alternative launch options."""
    
    print("\\n🔧 Alternative Launch Options:")
    print("=" * 40)
    
    print("\\n1. 🎮 Full GUI with Isaac Lab environment:")
    print("   python play_lerobot_g1_gui.py --enable_cameras")
    
    print("\\n2. 🎭 Headless with camera output saved:")
    print("   python play_lerobot_g1_gui.py --headless --enable_cameras")
    
    print("\\n3. 🚀 Direct G1 simulation (no library conflicts):")
    print("   python run_g1_inspire_direct.py")
    
    print("\\n4. 🧠 LeRobot integration demo:")
    print("   python lerobot_demo_complete.py")
    
    print("\\n5. 🔗 Real-time policy integration:")
    print("   python lerobot_g1_inspire_integration.py")
    
    print("\\n🎯 For best visualization experience:")
    print("   Use option 3 (run_g1_inspire_direct.py) which bypasses")
    print("   library conflicts while showing full Isaac Sim GUI")

def main():
    """Main function."""
    
    print("🎮 Isaac Sim Visualization Launcher")
    print("Choose how you want to launch the G1 Inspire simulation:")
    print()
    
    # Check system requirements
    print("🔍 System Check:")
    print(f"   Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f"   Python: {sys.executable}")
    print(f"   Working dir: {os.getcwd()}")
    print()
    
    # Show the main launch option
    success = launch_with_visualization()
    
    if success:
        # Show alternative options
        launch_alternative_options()
    
    print("\\n🎉 Ready to visualize G1 Inspire + LeRobot integration!")

if __name__ == "__main__":
    main()
