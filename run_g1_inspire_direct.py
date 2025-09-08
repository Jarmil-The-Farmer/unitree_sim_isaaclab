#!/usr/bin/env python3

"""
G1 Inspire + LeRobot Integration - Direct Subprocess Approach
This script runs the G1 Inspire simulation without library conflicts by using subprocess execution.
"""

import os
import sys
import subprocess
import time

def main():
    print("🤖 G1 Inspire Hands + LeRobot Integration")
    print("=" * 60)
    print("Running via subprocess to avoid library conflicts")
    print("=" * 60)
    
    # Check environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda environment: {conda_env}")
    
    if conda_env != 'unitree_sim_env':
        print("⚠️  Warning: Not in unitree_sim_env environment")
        print("Please run: conda activate unitree_sim_env")
    
    # Check if required files exist
    unitree_sim_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab"
    sim_main_path = os.path.join(unitree_sim_path, "sim_main.py")
    
    if not os.path.exists(sim_main_path):
        print(f"❌ Error: sim_main.py not found at {sim_main_path}")
        return False
    
    print(f"✓ Found unitree simulation at: {sim_main_path}")
    
    # Prepare command
    cmd = [
        sys.executable,
        sim_main_path,
        "--task", "Isaac-PickPlace-RedBlock-G129-Inspire-Joint",
        "--headless",
        "--enable_cameras",
        "--action_source", "policy",  # Use policy mode for LeRobot integration
        "--step_hz", "20"  # 20 Hz control frequency
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n🚀 Starting G1 Inspire simulation...")
    print("This will run the G1 robot with Inspire hands in the PickPlace environment")
    print("Press Ctrl+C to stop the simulation\n")
    
    try:
        # Set up environment for the subprocess
        env = os.environ.copy()
        env["PROJECT_ROOT"] = unitree_sim_path
        
        # Run the simulation
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env
        )
        
        print("Simulation started! Output:")
        print("-" * 40)
        
        # Stream output in real-time
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    
                    # Check for key indicators
                    if "robot control system started" in output:
                        print("\n🎉 ✓ Robot control system is running!")
                    elif "Environment device" in output:
                        print("✓ Environment initialized successfully!")
                    elif "ERROR" in output.upper():
                        print("⚠️  Error detected in simulation")
                        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping simulation...")
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                print("✓ Simulation stopped gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing simulation process")
                process.kill()
                process.wait()
                
        # Get final return code
        returncode = process.poll()
        if returncode == 0:
            print("\n✅ Simulation completed successfully!")
        elif returncode is None:
            print("\n⏸️  Simulation was interrupted")
        else:
            print(f"\n❌ Simulation ended with error code: {returncode}")
            
        return returncode == 0
        
    except FileNotFoundError:
        print(f"❌ Error: Python executable not found: {sys.executable}")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

if __name__ == "__main__":
    print("Direct G1 Inspire Simulation Runner")
    print("Bypasses Python library conflicts by using subprocess execution")
    print()
    
    success = main()
    
    if success:
        print("\n🎉 SUCCESS: G1 Inspire simulation completed!")
        print("\nNext steps:")
        print("1. The G1 robot with Inspire hands environment is working")
        print("2. You can now integrate LeRobot policies")
        print("3. Check the simulation output for any specific errors")
    else:
        print("\n❌ Simulation failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the unitree_sim_env conda environment")
        print("2. Check that Isaac Sim is properly installed")
        print("3. Verify unitree_sim_isaaclab is properly set up")
    
    sys.exit(0 if success else 1)
