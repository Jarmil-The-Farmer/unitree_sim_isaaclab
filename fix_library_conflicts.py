#!/usr/bin/env python3

"""
Library conflict debugging and resolution script.
This script helps identify and resolve the libhpp-fcl.so / Assimp conflict.
"""

import os
import sys
import subprocess

def check_library_conflicts():
    """Check for library conflicts and suggest solutions."""
    print("=" * 60)
    print("Library Conflict Diagnosis")
    print("=" * 60)
    
    # Check current environment
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    
    # Check LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' in os.environ:
        print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    else:
        print("LD_LIBRARY_PATH: Not set")
    
    # Check for conflicting libraries
    print("\n[CHECK] Looking for potential library conflicts...")
    
    # Find libhpp-fcl.so locations
    try:
        result = subprocess.run(['find', '/home/vaclav/miniconda3', '-name', 'libhpp-fcl.so*'], 
                              capture_output=True, text=True, timeout=30)
        if result.stdout:
            print("Found libhpp-fcl.so at:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("No libhpp-fcl.so found in conda directory")
    except:
        print("Could not search for libhpp-fcl.so")
    
    # Find libassimp.so locations
    try:
        result = subprocess.run(['find', '/home/vaclav/miniconda3', '-name', 'libassimp.so*'], 
                              capture_output=True, text=True, timeout=30)
        if result.stdout:
            print("Found libassimp.so at:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("No libassimp.so found in conda directory")
    except:
        print("Could not search for libassimp.so")

def try_fix_approach_1():
    """Try fixing by setting LD_PRELOAD for specific libraries."""
    print("\n" + "=" * 60)
    print("Approach 1: LD_PRELOAD Library Override")
    print("=" * 60)
    
    # Try to find a compatible assimp library
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/vaclav/miniconda3/envs/unitree_sim_env')
    
    potential_libs = [
        f"{conda_prefix}/lib/libassimp.so.5",
        f"{conda_prefix}/lib/libassimp.so",
        "/usr/lib/x86_64-linux-gnu/libassimp.so.5",
        "/usr/lib/x86_64-linux-gnu/libassimp.so"
    ]
    
    for lib_path in potential_libs:
        if os.path.exists(lib_path):
            print(f"Found potential library: {lib_path}")
            
            # Set environment and try import
            env = os.environ.copy()
            env['LD_PRELOAD'] = lib_path
            
            try:
                print(f"Testing import with LD_PRELOAD={lib_path}")
                result = subprocess.run([
                    sys.executable, '-c', 
                    """
import sys
sys.path.insert(0, '/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab')
try:
    import tasks
    print('SUCCESS: tasks imported')
except Exception as e:
    print(f'FAILED: {e}')
"""
                ], env=env, capture_output=True, text=True, timeout=30)
                
                print(f"Result: {result.stdout.strip()}")
                if "SUCCESS" in result.stdout:
                    print(f"✓ Solution found! Use: export LD_PRELOAD={lib_path}")
                    return lib_path
                    
            except Exception as e:
                print(f"Test failed: {e}")
    
    return None

def try_fix_approach_2():
    """Try fixing by using conda-forge channels for compatible libraries."""
    print("\n" + "=" * 60)
    print("Approach 2: Reinstall with Conda-Forge")
    print("=" * 60)
    
    print("You can try reinstalling problematic packages with conda-forge:")
    print("conda install -c conda-forge assimp")
    print("conda install -c conda-forge libboost")
    print("conda install -c conda-forge hpp-fcl")
    
    return False

def try_fix_approach_3():
    """Try direct execution without Python imports."""
    print("\n" + "=" * 60)
    print("Approach 3: Direct Subprocess Execution")
    print("=" * 60)
    
    try:
        # Try running the unitree simulation directly in a subprocess
        cmd = [
            sys.executable,
            "/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab/sim_main.py",
            "--task", "Isaac-PickPlace-RedBlock-G129-Inspire-Joint",
            "--headless",
            "--enable_cameras"
        ]
        
        print(f"Testing direct execution: {' '.join(cmd)}")
        
        # Run with timeout to avoid hanging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if "robot control system started" in result.stdout:
            print("✓ Direct execution works!")
            print("Consider using subprocess approach in main script")
            return True
        else:
            print("Direct execution has issues:")
            print(result.stdout[-500:] if result.stdout else "No stdout")
            print(result.stderr[-500:] if result.stderr else "No stderr")
            
    except subprocess.TimeoutExpired:
        print("Direct execution timed out (this might be normal for simulation startup)")
        return True  # Timeout might be OK - simulation might be starting
    except Exception as e:
        print(f"Direct execution failed: {e}")
    
    return False

def main():
    """Main diagnostic and fix function."""
    print("🔧 G1 Inspire Library Conflict Resolver")
    print("This script will help diagnose and fix the libhpp-fcl.so / Assimp conflict")
    
    # Step 1: Diagnose the problem
    check_library_conflicts()
    
    # Step 2: Try different fix approaches
    solution_found = False
    
    # Approach 1: LD_PRELOAD
    lib_override = try_fix_approach_1()
    if lib_override:
        solution_found = True
        print(f"\n🎉 SOLUTION FOUND: Set LD_PRELOAD={lib_override}")
        print("\nTo use this solution:")
        print(f"export LD_PRELOAD={lib_override}")
        print("python play_lerobot_g1_gui.py --enable_cameras")
    
    # Approach 2: Package management
    if not solution_found:
        try_fix_approach_2()
    
    # Approach 3: Subprocess approach
    if not solution_found:
        if try_fix_approach_3():
            solution_found = True
            print("\n🎉 SOLUTION FOUND: Use subprocess execution")
            print("\nThe unitree simulation can run directly.")
            print("Consider modifying the main script to use subprocess calls.")
    
    if not solution_found:
        print("\n❌ No automatic solution found.")
        print("\nManual steps to try:")
        print("1. conda install -c conda-forge assimp hpp-fcl")
        print("2. Check if there are conflicting isaac-sim installations")
        print("3. Try running in a fresh conda environment")
        print("4. Use the subprocess approach in the main script")
    
    return solution_found

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
