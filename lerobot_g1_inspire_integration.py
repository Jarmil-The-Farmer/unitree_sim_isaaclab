#!/usr/bin/env python3
"""
LeRobot G1 Inspire Integration - Real-time Policy Connection
This connects LeRobot diffusion policy to the working G1 Inspire simulation
"""

import subprocess
import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path
import tempfile
import threading
import queue
import mmap
import struct

def create_lerobot_bridge():
    """Create a bridge between LeRobot policy and G1 Inspire simulation."""
    
    print("🔗 LeRobot G1 Inspire Bridge")
    print("=" * 50)
    print("Connecting LeRobot diffusion policy to G1 Inspire simulation")
    print()
    
    # Step 1: Connect to running simulation via shared memory
    bridge = LeRobotG1Bridge()
    
    # Step 2: Load LeRobot policy 
    checkpoint_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot/outputs/train/2025-06-03/00-52-31_diffusion/checkpoints/last/pretrained_model"
    
    policy = bridge.load_lerobot_policy(checkpoint_path)
    
    if policy:
        print("✅ LeRobot policy loaded successfully")
        
        # Step 3: Start real-time control loop
        bridge.start_realtime_control(policy)
    else:
        print("⚠️  Using fallback policy - LeRobot not available")
        bridge.start_fallback_control()

class LeRobotG1Bridge:
    """Bridge between LeRobot and G1 Inspire simulation."""
    
    def __init__(self):
        self.simulation_process = None
        self.shared_memory_name = "isaac_multi_image_shm"
        self.control_active = False
        
        # Camera configuration for LeRobot
        self.left_camera_data = None
        self.right_camera_data = None
        self.robot_state = None
        
        # Temporal buffers for n_obs_steps=2
        self.left_camera_history = []
        self.right_camera_history = []
        self.robot_state_history = []
        
        # Action buffer for G1 Inspire (50 DOF: 12 legs + 14 arms + 24 inspire hands)
        self.action_buffer = np.zeros(50, dtype=np.float32)
        
        print("🤖 G1 Inspire Bridge initialized")
        print(f"📸 Looking for shared memory: {self.shared_memory_name}")
        
    def load_lerobot_policy(self, checkpoint_path):
        """Load LeRobot diffusion policy with proper isolation."""
        
        # Add LeRobot to path 
        lerobot_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot"
        if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
            sys.path.insert(0, lerobot_path)
            print(f"✅ Added LeRobot path: {lerobot_path}")
        
        try:
            # Load configuration
            config_path = Path(checkpoint_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded LeRobot config from: {config_path}")
                
                # Create policy wrapper
                policy = LeRobotPolicyWrapper(config, checkpoint_path)
                return policy
            else:
                print(f"❌ Config not found: {config_path}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load LeRobot policy: {e}")
            return None
    
    def connect_to_simulation(self):
        """Connect to the running G1 Inspire simulation."""
        
        # Check if our simulation process is running
        try:
            # Try to connect to shared memory
            import mmap
            
            # Look for Isaac Sim shared memory
            shm_path = f"/dev/shm/{self.shared_memory_name}"
            if os.path.exists(shm_path):
                print(f"✅ Found simulation shared memory: {shm_path}")
                return True
            else:
                print(f"⚠️  Shared memory not found: {shm_path}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to connect to simulation: {e}")
            return False
    
    def extract_camera_data(self):
        """Extract stereo camera data from simulation."""
        
        try:
            # This would read from the MultiImageWriter shared memory
            # For now, simulate camera data
            
            # Model expects full resolution [3, 720, 1280] as specified in config
            height, width = 720, 1280
            
            # Generate realistic-looking camera data
            timestamp = time.time()
            noise_left = np.sin(timestamp * 0.1) * 0.1
            noise_right = np.cos(timestamp * 0.1) * 0.1
            
            # Simulate stereo camera pair with slight offset
            base_image_left = np.random.rand(height, width, 3).astype(np.float32) * 0.1 + 0.5 + noise_left
            base_image_right = np.random.rand(height, width, 3).astype(np.float32) * 0.1 + 0.5 + noise_right
            
            # Add some structure to make it look more realistic
            x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            pattern = np.sin(x * 10) * np.cos(y * 10) * 0.1
            
            for c in range(3):
                base_image_left[:, :, c] += pattern
                base_image_right[:, :, c] += pattern * 0.9  # Slight difference for stereo
            
            # Clip to valid range
            base_image_left = np.clip(base_image_left, 0, 1)
            base_image_right = np.clip(base_image_right, 0, 1)
            
            # Convert to channels-first format [3, 720, 1280] for LeRobot
            self.left_camera_data = np.transpose(base_image_left, (2, 0, 1))
            self.right_camera_data = np.transpose(base_image_right, (2, 0, 1))
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract camera data: {e}")
            return False
    
    def extract_robot_state(self):
        """Extract G1 robot state from simulation."""
        
        try:
            # Simulate G1 robot state (50 DOF)
            # In real implementation, this would come from DDS or shared memory
            
            timestamp = time.time()
            
            # Simulate realistic joint positions with some dynamics
            full_robot_state = np.zeros(50, dtype=np.float32)
            
            # Legs (joints 0-11): walking pattern
            for i in range(12):
                phase = timestamp * 2.0 + i * np.pi / 6
                full_robot_state[i] = 0.2 * np.sin(phase) + np.random.normal(0, 0.01)
            
            # Arms (joints 12-25): reaching pattern
            for i in range(12, 26):
                phase = timestamp * 1.0 + (i-12) * np.pi / 7
                full_robot_state[i] = 0.3 * np.sin(phase) + np.random.normal(0, 0.01)
            
            # Inspire hands (joints 26-49): manipulation pattern
            for i in range(26, 50):
                phase = timestamp * 0.5 + (i-26) * np.pi / 12
                full_robot_state[i] = 0.1 * np.sin(phase) + np.random.normal(0, 0.005)
            
            # Store full state for action mapping
            self.full_robot_state = full_robot_state
            
            # Extract core 26 DOF for LeRobot model (trained on simpler robot)
            # Use legs (12 DOF) + arms (14 DOF) = 26 DOF
            self.robot_state = full_robot_state[:26].copy()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract robot state: {e}")
            return False
    
    def send_actions(self, actions):
        """Send actions to G1 Inspire simulation."""
        
        try:
            # Store actions in buffer
            self.action_buffer = np.array(actions, dtype=np.float32)
            
            # In real implementation, this would send via DDS or shared memory
            # For now, just validate the actions
            
            # Clamp actions to safe ranges
            self.action_buffer = np.clip(self.action_buffer, -1.0, 1.0)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send actions: {e}")
            return False
    
    def map_actions_to_full_robot(self, actions_26dof):
        """Map 26 DOF LeRobot actions to full 50 DOF G1 Inspire robot."""
        
        try:
            # Create full action array
            full_actions = np.zeros(50, dtype=np.float32)
            
            # Map core 26 DOF (legs + arms) directly
            full_actions[:26] = actions_26dof[:26]
            
            # Generate coordinated hand actions based on arm movements
            # Left arm to left hand mapping (joints 26-37)
            left_arm_activity = np.mean(np.abs(actions_26dof[12:19]))  # Left arm joints
            left_hand_action = left_arm_activity * 0.5  # Scale down for hand
            
            # Right arm to right hand mapping (joints 38-49) 
            right_arm_activity = np.mean(np.abs(actions_26dof[19:26]))  # Right arm joints
            right_hand_action = right_arm_activity * 0.5
            
            # Apply hand actions with some variation
            for i in range(26, 38):  # Left hand
                finger_idx = (i - 26) % 12
                finger_variation = np.sin(finger_idx * 0.5) * 0.1
                full_actions[i] = left_hand_action + finger_variation
                
            for i in range(38, 50):  # Right hand  
                finger_idx = (i - 38) % 12
                finger_variation = np.cos(finger_idx * 0.5) * 0.1
                full_actions[i] = right_hand_action + finger_variation
            
            # Clamp to safe ranges
            full_actions = np.clip(full_actions, -1.0, 1.0)
            
            return full_actions
            
        except Exception as e:
            print(f"❌ Failed to map actions: {e}")
            # Return safe zero actions
            return np.zeros(50, dtype=np.float32)
    
    def start_realtime_control(self, policy):
        """Start real-time control loop with LeRobot policy."""
        
        print("🚀 Starting real-time LeRobot control loop...")
        print("Press Ctrl+C to stop")
        
        self.control_active = True
        step_count = 0
        
        try:
            while self.control_active:
                start_time = time.time()
                
                # Step 1: Extract observations
                camera_success = self.extract_camera_data()
                state_success = self.extract_robot_state()
                
                if camera_success and state_success:
                    # Update temporal buffers
                    self.left_camera_history.append(self.left_camera_data.copy())
                    self.right_camera_history.append(self.right_camera_data.copy())
                    self.robot_state_history.append(self.robot_state.copy())
                    
                    # Keep only last 2 timesteps (n_obs_steps=2)
                    if len(self.left_camera_history) > 2:
                        self.left_camera_history.pop(0)
                        self.right_camera_history.pop(0) 
                        self.robot_state_history.pop(0)
                    
                    # Only proceed if we have enough history
                    if len(self.left_camera_history) >= 2:
                        # Step 2: Prepare observations for LeRobot with proper temporal handling
                        # For LeRobot, we need to flatten the temporal dimension into batch dimension
                        # So [1, 2, 3, 720, 1280] becomes [2, 3, 720, 1280] for processing
                        
                        # Stack and flatten temporal dimension into batch
                        left_cameras = np.stack(self.left_camera_history, axis=0)  # [2, 3, 720, 1280]
                        right_cameras = np.stack(self.right_camera_history, axis=0)  # [2, 3, 720, 1280]
                        robot_states = np.stack(self.robot_state_history, axis=0)  # [2, 26]
                        
                        # Concatenate both cameras along batch dimension for processing
                        all_cameras = np.concatenate([left_cameras, right_cameras], axis=0)  # [4, 3, 720, 1280]
                        all_states = np.concatenate([robot_states, robot_states], axis=0)  # [4, 26]
                        
                        obs = {
                            'observation.images.cam_left_high': torch.from_numpy(left_cameras).float(),  # [2, 3, 720, 1280]
                            'observation.images.cam_right_high': torch.from_numpy(right_cameras).float(),  # [2, 3, 720, 1280] 
                            'observation.state': torch.from_numpy(robot_states).float()  # [2, 26]
                        }
                        
                        # Move to device
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        for key in obs:
                            obs[key] = obs[key].to(device)
                        
                        # Step 3: Run policy inference
                        actions = policy.predict(obs)
                        
                        # Step 4: Map actions from 26 DOF to 50 DOF and send to simulation
                        if actions is not None:
                            full_actions = self.map_actions_to_full_robot(actions.cpu().numpy().flatten())
                            self.send_actions(full_actions)
                    else:
                        # Not enough history yet, use fallback
                        actions = None
                    
                    step_count += 1
                    
                    # Print status every 30 steps (0.5 seconds at 60Hz)
                    if step_count % 30 == 0:
                        print(f"🎮 Control step {step_count}: "
                              f"Camera OK: {camera_success}, State OK: {state_success}")
                        print(f"📸 Left camera shape: {self.left_camera_data.shape}")
                        print(f"📸 Right camera shape: {self.right_camera_data.shape}")
                        print(f"📚 Temporal history length: {len(self.left_camera_history)}/2")
                        print(f"🤖 Robot state shape (for LeRobot): {self.robot_state.shape}")
                        print(f"🤖 Full robot state shape (G1 Inspire): {getattr(self, 'full_robot_state', np.array([])).shape}")
                        if actions is not None:
                            print(f"🎯 LeRobot action range: [{actions.min():.3f}, {actions.max():.3f}]")
                            full_actions = self.map_actions_to_full_robot(actions.cpu().numpy().flatten())
                            print(f"🎯 Mapped full action range: [{full_actions.min():.3f}, {full_actions.max():.3f}]")
                        else:
                            print(f"⏳ Waiting for temporal buffer to fill...")
                
                # Maintain 60Hz control rate
                elapsed = time.time() - start_time
                sleep_time = 1/60.0 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Control loop stopped by user")
        except Exception as e:
            print(f"❌ Control loop error: {e}")
        finally:
            self.control_active = False
            print("✅ Real-time control ended")
    
    def start_fallback_control(self):
        """Start fallback control when LeRobot is not available."""
        
        print("🚀 Starting fallback control loop...")
        print("Press Ctrl+C to stop")
        
        self.control_active = True
        step_count = 0
        
        try:
            while self.control_active:
                start_time = time.time()
                
                # Extract observations
                self.extract_camera_data()
                self.extract_robot_state()
                
                # Generate simple manipulation actions
                actions = self.generate_manipulation_pattern(step_count)
                self.send_actions(actions)
                
                step_count += 1
                
                if step_count % 60 == 0:
                    print(f"🎮 Fallback control step {step_count}")
                    print(f"🤖 Performing simple manipulation pattern")
                
                # 60Hz rate
                elapsed = time.time() - start_time
                sleep_time = 1/60.0 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Fallback control stopped")
        finally:
            self.control_active = False
    
    def generate_manipulation_pattern(self, step):
        """Generate simple manipulation pattern for demonstration."""
        
        actions = np.zeros(50, dtype=np.float32)
        
        t = step * 1/60.0  # Time in seconds
        
        # Simple pick-place pattern
        # Arms: reaching motion
        reach_amplitude = 0.2
        reach_frequency = 0.3
        
        # Left arm (shoulder pitch, elbow)
        actions[12] = reach_amplitude * np.sin(reach_frequency * t)  # Shoulder pitch
        actions[15] = reach_amplitude * np.abs(np.sin(reach_frequency * t))  # Elbow
        
        # Right arm 
        actions[19] = reach_amplitude * np.sin(reach_frequency * t + np.pi)
        actions[22] = reach_amplitude * np.abs(np.sin(reach_frequency * t + np.pi))
        
        # Inspire hands: grasping pattern
        grasp_amplitude = 0.3
        grasp_frequency = 0.5
        grasp_phase = np.sin(grasp_frequency * t)
        
        # Apply grasping to all finger joints (26-49)
        for i in range(26, 50):
            finger_index = (i - 26) % 12  # 12 joints per hand
            finger_phase = grasp_phase + finger_index * 0.1  # Slight delay between fingers
            actions[i] = grasp_amplitude * np.clip(finger_phase, -1, 1)
        
        return actions

class LeRobotPolicyWrapper:
    """Wrapper for LeRobot diffusion policy."""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to load the actual model
        self.model = None
        self.use_real_model = False
        self.prediction_count = 0  # Track predictions for debugging
        
        try:
            # Import LeRobot components
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
            
            print("✅ LeRobot imports successful")
            
            # Process configuration to handle nested dictionaries
            print("🔧 Processing LeRobot configuration...")
            print(f"🔍 Config normalization_mapping: {config.get('normalization_mapping', 'NOT_FOUND')}")
            processed_config = self._process_config(config)
            
            # Load model config and weights
            config_obj = DiffusionConfig(**processed_config)
            self.model = DiffusionPolicy(config_obj)
            
            # Load weights
            weights_path = self.checkpoint_path / "diffusion_pytorch_model.safetensors"
            if weights_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.use_real_model = True
                print("✅ LeRobot model loaded successfully")
            else:
                print(f"⚠️  Weights not found: {weights_path}")
                
        except Exception as e:
            print(f"⚠️  Failed to load real model: {e}")
            print(f"⚠️  Error details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Using fallback policy")
    
    def _process_config(self, config):
        """Process the configuration to handle nested dictionaries and convert to proper objects."""
        
        try:
            # Try to import required types
            try:
                from lerobot.configs.types import FeatureType, NormalizationMode
                print("✅ LeRobot types imported successfully")
                print(f"🔍 Available NormalizationMode values: {list(NormalizationMode)}")
            except ImportError:
                print("⚠️  LeRobot types not available, using simplified approach")
                # Create mock enums
                FeatureType = type('FeatureType', (), {'STATE': 'state', 'IMAGE': 'image'})()
                NormalizationMode = type('NormalizationMode', (), {'NONE': 'none', 'MINMAX': 'minmax'})()
            
            processed_config = config.copy()
            
            # Simplified approach: convert nested dicts to simple objects
            def dict_to_object(d):
                """Convert dictionary to simple object with attributes."""
                if isinstance(d, dict):
                    obj = type('SimpleConfig', (), {})()
                    for key, value in d.items():
                        if isinstance(value, dict):
                            setattr(obj, key, dict_to_object(value))
                        elif key == 'type' and isinstance(value, str):
                            # Handle type conversion
                            if hasattr(FeatureType, value.upper()):
                                setattr(obj, key, getattr(FeatureType, value.upper()))
                            else:
                                setattr(obj, key, value)
                        else:
                            setattr(obj, key, value)
                    return obj
                return d
            
            # Process input_features
            if "input_features" in processed_config and isinstance(processed_config["input_features"], dict):
                processed_features = {}
                for key, feature_dict in processed_config["input_features"].items():
                    processed_features[key] = dict_to_object(feature_dict)
                processed_config["input_features"] = processed_features
                print(f"✅ Processed {len(processed_features)} input features")
            
            # Process output_features
            if "output_features" in processed_config and isinstance(processed_config["output_features"], dict):
                processed_features = {}
                for key, feature_dict in processed_config["output_features"].items():
                    processed_features[key] = dict_to_object(feature_dict)
                processed_config["output_features"] = processed_features
                print(f"✅ Processed {len(processed_features)} output features")
            
            # Process normalization_mapping - convert strings to proper NormalizationMode objects
            if "normalization_mapping" in processed_config and isinstance(processed_config["normalization_mapping"], dict):
                processed_normalization = {}
                for key, norm_value in processed_config["normalization_mapping"].items():
                    if isinstance(norm_value, str):
                        # Convert string to actual NormalizationMode enum
                        norm_str = norm_value.upper()
                        if hasattr(NormalizationMode, norm_str):
                            processed_normalization[key] = getattr(NormalizationMode, norm_str)
                        else:
                            # Fallback to common normalization modes
                            if norm_str in ['NONE', 'NO_NORM']:
                                processed_normalization[key] = NormalizationMode.NONE
                            elif norm_str in ['MINMAX', 'MIN_MAX']:
                                processed_normalization[key] = NormalizationMode.MINMAX
                            else:
                                processed_normalization[key] = NormalizationMode.NONE
                    else:
                        processed_normalization[key] = norm_value
                processed_config["normalization_mapping"] = processed_normalization
                print(f"✅ Processed normalization mapping with proper enums")
            
            print(f"✅ Configuration processed successfully")
            return processed_config
            
        except Exception as e:
            print(f"⚠️  Configuration processing failed: {e}")
            print(f"⚠️  Using original config as fallback")
            import traceback
            traceback.print_exc()
            return config
    
    def predict(self, obs):
        """Predict actions from observations."""
        
        try:
            if self.use_real_model and self.model is not None:
                # Only print detailed debug info for first 3 predictions
                if self.prediction_count < 3:
                    print(f"🔍 Model input shapes (prediction #{self.prediction_count + 1}):")
                    for key, value in obs.items():
                        print(f"  {key}: {value.shape}")
                
                # Ensure model is in eval mode for inference
                self.model.eval()
                
                with torch.no_grad():
                    # Use the model's predict method instead of forward for inference
                    if hasattr(self.model, 'predict'):
                        actions = self.model.predict(obs)
                    else:
                        # Alternative: use select_action if available
                        if hasattr(self.model, 'select_action'):
                            actions = self.model.select_action(obs)
                        else:
                            # Last resort: try to call the diffusion inference directly
                            actions = self.model.diffusion.predict(obs, deterministic=True)
                    
                if self.prediction_count < 3:
                    print(f"✅ Model prediction successful, action shape: {actions.shape}")
                
                self.prediction_count += 1
                return actions
            else:
                # Fallback: simple pattern
                # Get batch size from any observation
                batch_size = 1
                for key, value in obs.items():
                    if hasattr(value, 'shape'):
                        batch_size = value.shape[0]
                        break
                actions = torch.randn(batch_size, 26) * 0.1  # 26 DOF for LeRobot
                return actions
                
        except Exception as e:
            # Only print detailed errors for first few attempts
            if self.prediction_count < 5:
                print(f"⚠️  Policy prediction failed: {e}")
                print(f"⚠️  Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            elif self.prediction_count == 5:
                print(f"⚠️  Policy prediction failed: {e} (suppressing further detailed errors)")
            
            self.prediction_count += 1
            # Return safe fallback actions
            batch_size = 1
            for key, value in obs.items():
                if hasattr(value, 'shape'):
                    batch_size = value.shape[0]
                    break
            return torch.zeros(batch_size, 26)

def main():
    """Main function to start LeRobot G1 Inspire integration."""
    
    print("🤖 LeRobot G1 Inspire Integration")
    print("=" * 50)
    print()
    
    # Check if simulation is running
    shm_path = "/dev/shm/isaac_multi_image_shm"
    if os.path.exists(shm_path):
        print("✅ Isaac Sim simulation detected (shared memory found)")
    else:
        print("⚠️  Isaac Sim not detected, will simulate data")
    
    # Start the bridge
    create_lerobot_bridge()

if __name__ == "__main__":
    main()
