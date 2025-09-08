#!/usr/bin/env python3

"""Script to play a Lerobot diffusion policy with Unitree G1 in Isaac Lab with GUI.

This script integrates LeRobot's diffusion policy with Isaac Lab's Unitree G1 robot simulation.
It handles the action space mapping from LeRobot's 26D actions to Isaac Lab's 37D joint space.

Key features:
- Loads pretrained LeRobot diffusion models for humanoid control
- Maps 5-finger hand actions to 3-finger G1 hand configuration  
- Provides fallback to simple walking pattern when LeRobot is unavailable
"""

import argparse
import json
import os
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Environment isolation for LeRobot
def ensure_lerobot_available():
    """Ensure LeRobot is available with proper environment isolation."""
    import sys
    lerobot_paths = [
        "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot"
    ]
    
    for path in lerobot_paths:
        lerobot_init = os.path.join(path, "lerobot", "__init__.py")
        if os.path.exists(lerobot_init):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"✓ LeRobot available at: {lerobot_init}")
            print("✓ LeRobot loaded via environment isolation")
            return True
    
    print("[WARNING] LeRobot not found in expected locations")
    print("[WARNING] Using placeholder policy instead")
    return False

# Initialize LeRobot availability
LEROBOT_AVAILABLE = ensure_lerobot_available()

from isaaclab.app import AppLauncher

# Add argparse arguments for AppLauncher
parser = argparse.ArgumentParser(description="Play Lerobot policy with Unitree G1 (with GUI).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint_path", type=str, default="/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot/outputs/train/2025-06-03/00-52-31_diffusion/checkpoints/last/pretrained_model", help="Path to Lerobot checkpoint directory.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time if possible.")
# Note: --device is provided by AppLauncher, so we don't add it here
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the Isaac Sim app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching the app
from isaaclab.utils.dict import print_dict

class LerobotPolicy:
    """A wrapper for Lerobot diffusion policy."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        # Load configuration
        config_path = self.checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"[INFO] Loaded Lerobot config from: {config_path}")
        print_dict(self.config, nesting=2)
        
        # Initialize diffusion model if LeRobot is available
        self.use_real_model = LEROBOT_AVAILABLE
        if self.use_real_model:
            try:
                # Re-ensure LeRobot path is available after Isaac Lab initialization
                import sys
                lerobot_path = "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot"
                if lerobot_path not in sys.path:
                    sys.path.insert(0, lerobot_path)
                    print(f"[INFO] Re-added LeRobot path after Isaac Lab init: {lerobot_path}")
                
                # Also ensure the parent path is available for imports
                parent_path = "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot"
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
                    print(f"[INFO] Added LeRobot parent path: {parent_path}")
                
                # Load the real diffusion policy using the same method as play_lerobot_g1_final.py
                from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
                from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
                from lerobot.configs.types import FeatureType, NormalizationMode
                from safetensors.torch import load_file
                
                # Use the exact same approach as the working final script
                print(f"[INFO] Loading DiffusionPolicy using manual config creation...")
                
                # Create a simple FeatureConfig-like class to replace dicts
                class SimpleFeatureConfig:
                    def __init__(self, type, shape, **kwargs):
                        # Convert string type to FeatureType enum
                        if isinstance(type, str):
                            if type == "STATE":
                                self.type = FeatureType.STATE  # Keep STATE as STATE for robot_state_feature
                            elif type == "VISUAL":
                                self.type = FeatureType.VISUAL
                            elif type == "ACTION":
                                self.type = FeatureType.ACTION
                            else:
                                self.type = FeatureType(type)  # Try to create enum from string
                        else:
                            self.type = type
                        
                        self.shape = shape
                        # Ensure shape is a list, not tuple
                        if isinstance(self.shape, tuple):
                            self.shape = list(self.shape)
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                    
                    def __repr__(self):
                        return f"FeatureConfig(type={self.type}, shape={self.shape})"
                
                # Convert dict features to proper FeatureConfig objects
                print(f"[INFO] Converting feature configurations...")
                processed_config = self.config.copy()
                
                # Process input_features to convert dicts to FeatureConfig objects
                if "input_features" in processed_config:
                    input_features = {}
                    for key, feature_dict in processed_config["input_features"].items():
                        input_features[key] = SimpleFeatureConfig(**feature_dict)
                    processed_config["input_features"] = input_features
                
                # Process output_features to convert dicts to FeatureConfig objects  
                if "output_features" in processed_config:
                    output_features = {}
                    for key, feature_dict in processed_config["output_features"].items():
                        output_features[key] = SimpleFeatureConfig(**feature_dict)
                    processed_config["output_features"] = output_features
                
                # Process normalization_mapping to convert strings to NormalizationMode enums
                if "normalization_mapping" in processed_config:
                    norm_mapping = {}
                    for key, norm_str in processed_config["normalization_mapping"].items():
                        if isinstance(norm_str, str):
                            norm_mapping[key] = NormalizationMode(norm_str)
                        else:
                            norm_mapping[key] = norm_str
                    processed_config["normalization_mapping"] = norm_mapping
                
                print(f"[DEBUG] Processed config with proper feature objects and normalization enums")
                # Create config using DiffusionConfig constructor directly
                config = DiffusionConfig(**processed_config)
                self.model = DiffusionPolicy(config)
                
                # Load weights manually
                checkpoint_file = self.checkpoint_path / "diffusion_pytorch_model.safetensors"
                if checkpoint_file.exists():
                    state_dict = load_file(str(checkpoint_file))
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.eval()
                    self.model.to(self.device)
                    print(f"[INFO] Successfully loaded real G1 diffusion model from checkpoint: {checkpoint_file}")
                else:
                    print(f"[WARNING] Checkpoint file not found: {checkpoint_file}")
                    print(f"[WARNING] Falling back to placeholder policy")
                    self.use_real_model = False
                
            except Exception as e:
                import sys
                print(f"[WARNING] Failed to load LeRobot model: {e}")
                print(f"[DEBUG] Current sys.path contains LeRobot: {any('lerobot' in p for p in sys.path)}")
                print(f"[DEBUG] LeRobot path exists: {os.path.exists('/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot')}")
                
                # Try simpler approach - use the working method from play_lerobot_g1_final.py
                try:
                    print(f"[INFO] Trying exact method from working final script...")
                    
                    # Force re-add LeRobot paths
                    import sys
                    lerobot_paths = [
                        "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot",
                        "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot"
                    ]
                    for path in lerobot_paths:
                        if path not in sys.path:
                            sys.path.insert(0, path)
                            print(f"[INFO] Force-added path: {path}")
                    
                    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
                    from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
                    from lerobot.configs.types import FeatureType, NormalizationMode
                    from safetensors.torch import load_file
                    
                    # Create a simple FeatureConfig-like class to replace dicts
                    class SimpleFeatureConfig:
                        def __init__(self, type, shape, **kwargs):
                            # Convert string type to FeatureType enum
                            if isinstance(type, str):
                                if type == "STATE":
                                    self.type = FeatureType.STATE  # Keep STATE as STATE for robot_state_feature
                                elif type == "VISUAL":
                                    self.type = FeatureType.VISUAL
                                elif type == "ACTION":
                                    self.type = FeatureType.ACTION
                                else:
                                    self.type = FeatureType(type)  # Try to create enum from string
                            else:
                                self.type = type
                            
                            self.shape = shape
                            # Ensure shape is a list, not tuple
                            if isinstance(self.shape, tuple):
                                self.shape = list(self.shape)
                            for k, v in kwargs.items():
                                setattr(self, k, v)
                        
                        def __repr__(self):
                            return f"FeatureConfig(type={self.type}, shape={self.shape})"
                    
                    # The working approach: manual config + manual weight loading
                    print(f"[INFO] Creating DiffusionConfig manually with proper feature objects...")
                    
                    # Process the config to convert dicts to proper FeatureConfig objects
                    processed_config = self.config.copy()
                    
                    # Process input_features to convert dicts to FeatureConfig objects
                    if "input_features" in processed_config:
                        input_features = {}
                        for key, feature_dict in processed_config["input_features"].items():
                            input_features[key] = SimpleFeatureConfig(**feature_dict)
                        processed_config["input_features"] = input_features
                    
                    # Process output_features to convert dicts to FeatureConfig objects  
                    if "output_features" in processed_config:
                        output_features = {}
                        for key, feature_dict in processed_config["output_features"].items():
                            output_features[key] = SimpleFeatureConfig(**feature_dict)
                        processed_config["output_features"] = output_features
                    
                    # Process normalization_mapping to convert strings to NormalizationMode enums
                    if "normalization_mapping" in processed_config:
                        norm_mapping = {}
                        for key, norm_str in processed_config["normalization_mapping"].items():
                            if isinstance(norm_str, str):
                                norm_mapping[key] = NormalizationMode(norm_str)
                            else:
                                norm_mapping[key] = norm_str
                        processed_config["normalization_mapping"] = norm_mapping
                    
                    print(f"[DEBUG] Created proper feature config objects and normalization enums")
                    config = DiffusionConfig(**processed_config)
                    print(f"[INFO] Creating DiffusionPolicy with config...")
                    self.model = DiffusionPolicy(config)
                    
                    # Load weights manually
                    checkpoint_file = self.checkpoint_path / "diffusion_pytorch_model.safetensors"
                    if checkpoint_file.exists():
                        print(f"[INFO] Loading weights from safetensors file...")
                        state_dict = load_file(str(checkpoint_file))
                        self.model.load_state_dict(state_dict, strict=False)
                        self.model.eval()
                        self.model.to(self.device)
                        print(f"[INFO] Successfully loaded real G1 diffusion model via manual method!")
                        return  # Success, exit the except block
                    else:
                        print(f"[WARNING] Checkpoint file not found: {checkpoint_file}")
                        
                except Exception as e2:
                    print(f"[WARNING] Manual method also failed: {e2}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[WARNING] Falling back to placeholder policy")
                self.use_real_model = False
        
        # Extract dimensions from config
        self.action_dim = self.config["output_features"]["action"]["shape"][0]
        self.state_dim = self.config["input_features"]["observation.state"]["shape"][0]
        
        print(f"[INFO] Action dimension: {self.action_dim}")
        print(f"[INFO] State dimension: {self.state_dim}")
        print(f"[INFO] Using real model: {self.use_real_model}")
        
        # Initialize observation history for temporal consistency
        self.obs_history = []
        self.n_obs_steps = self.config["n_obs_steps"]
        
        # Action mapping from 26D LeRobot to 37D Isaac Lab
        # Based on G1 joint structure: legs(12) + arms(12) + torso(2) + fingers(11)
        self.action_mapping = self._create_action_mapping()
        
        # Initialize some basic movement patterns for demonstration (fallback)
        self.time_step = 0
        
    def _create_action_mapping(self):
        """Create mapping from 26D LeRobot actions to 37D Isaac Lab actions."""
        # G1 joint order in Isaac Lab (37 joints):
        # Legs: left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll (6)
        #       right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll (6)
        # Torso: torso_yaw, torso_pitch (2)
        # Arms: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_elbow_roll (5)
        #       right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_elbow_roll (5)
        # Hands: left_hand fingers (varies, but in Isaac Lab it's 11 total finger joints)
        
        # LeRobot was trained with 26 actions, likely excluding some finger joints
        # We'll map the first 26 actions directly and handle fingers separately
        mapping = {}
        
        # Direct mapping for the first 26 joints (legs + torso + arms + some hand)
        for i in range(min(26, 37)):
            mapping[i] = i
            
        # For the remaining joints (fingers), we'll use default values or simple mapping
        # Since LeRobot was trained with 5-finger hands but G1 has 3-finger hands,
        # we need to adapt the finger actions
        
        return mapping
    
    def _map_finger_actions(self, lerobot_actions):
        """Map 5-finger LeRobot actions to 3-finger G1 actions.
        
        The pretrained LeRobot model was trained with 5-finger hands, but the actual
        Unitree G1 robot (via unitree_sdk2_python) has 3-finger hands. This function
        handles the mapping between these different finger configurations.
        
        Args:
            lerobot_actions: Actions from LeRobot model (26D)
            
        Returns:
            finger_actions: Mapped finger actions for Isaac Lab G1 (11D finger joints)
        """
        # This function handles the discrepancy between training data (5 fingers)
        # and the actual G1 robot (3 fingers per hand)
        
        batch_size = lerobot_actions.shape[0]
        
        # Isaac Lab G1 finger joint structure (11 total finger joints):
        # Left hand: thumb(2), index(2), middle(1) = 5 joints
        # Right hand: thumb(2), index(2), middle(1) = 5 joints  
        # Additional: 1 joint (possibly gripper or wrist)
        
        finger_actions = torch.zeros((batch_size, 11), device=self.device)
        
        # Strategy for 5-finger to 3-finger mapping:
        # 1. Map thumb directly (most important for grasping)
        # 2. Combine index+middle finger actions for index finger
        # 3. Use ring finger action for middle finger
        # 4. Ignore pinky finger action
        
        if lerobot_actions.shape[1] >= 26:
            for i in range(batch_size):
                # Assuming finger actions are encoded in the last part of the 26D action space
                # This mapping might need adjustment based on the actual LeRobot action structure
                
                # Left hand mapping (joints 0-4 in finger_actions)
                # Extract potential finger actions from LeRobot (speculative indices)
                if lerobot_actions.shape[1] > 16:  # Assuming joints 16-25 might include hand actions
                    # Left thumb (2 joints): directly map from LeRobot thumb actions
                    finger_actions[i, 0] = lerobot_actions[i, 16] if lerobot_actions.shape[1] > 16 else 0.0
                    finger_actions[i, 1] = lerobot_actions[i, 17] if lerobot_actions.shape[1] > 17 else 0.0
                    
                    # Left index (2 joints): combine index and middle from 5-finger model
                    left_index_action = (lerobot_actions[i, 18] + lerobot_actions[i, 19]) * 0.5 if lerobot_actions.shape[1] > 19 else 0.0
                    finger_actions[i, 2] = left_index_action
                    finger_actions[i, 3] = left_index_action * 0.8  # Slightly different for realism
                    
                    # Left middle (1 joint): use ring finger action from 5-finger model
                    finger_actions[i, 4] = lerobot_actions[i, 20] if lerobot_actions.shape[1] > 20 else 0.0
                
                # Right hand mapping (joints 5-9 in finger_actions)
                if lerobot_actions.shape[1] > 21:
                    # Right thumb (2 joints)
                    finger_actions[i, 5] = lerobot_actions[i, 21] if lerobot_actions.shape[1] > 21 else 0.0
                    finger_actions[i, 6] = lerobot_actions[i, 22] if lerobot_actions.shape[1] > 22 else 0.0
                    
                    # Right index (2 joints)
                    right_index_action = (lerobot_actions[i, 23] + lerobot_actions[i, 24]) * 0.5 if lerobot_actions.shape[1] > 24 else 0.0
                    finger_actions[i, 7] = right_index_action
                    finger_actions[i, 8] = right_index_action * 0.8
                    
                    # Right middle (1 joint)
                    finger_actions[i, 9] = lerobot_actions[i, 25] if lerobot_actions.shape[1] > 25 else 0.0
                
                # Additional joint (could be wrist or gripper)
                finger_actions[i, 10] = 0.0  # Set to neutral for now
        
        # Clamp finger actions to reasonable ranges
        finger_actions = torch.clamp(finger_actions, -1.0, 1.0)
        
        return finger_actions
        
    def __call__(self, obs_dict):
        """
        Predict actions given observations.
        
        Args:
            obs_dict: Dictionary containing observations from Isaac Lab environment
            
        Returns:
            actions: Tensor of actions for Isaac Lab (37 dimensions)
        """
        batch_size = obs_dict["policy"].shape[0]
        
        if self.use_real_model:
            # Use the actual LeRobot diffusion model
            try:
                # Extract state observations (need to map from Isaac Lab's 123D to LeRobot's 26D)
                isaac_state = obs_dict["policy"]  # 123D observation
                lerobot_state = self._map_isaac_to_lerobot_state(isaac_state)
                
                # Prepare observation for the model
                # Note: Real implementation would also include camera observations
                observation = {
                    "observation.state": lerobot_state,
                    # Camera observations would be added here if available
                    # "observation.images.cam_left_high": left_camera_tensor,
                    # "observation.images.cam_right_high": right_camera_tensor,
                }
                
                # Update observation history for temporal consistency
                self.obs_history.append(observation)
                if len(self.obs_history) > self.n_obs_steps:
                    self.obs_history.pop(0)
                
                # Pad observation history if needed
                while len(self.obs_history) < self.n_obs_steps:
                    self.obs_history.insert(0, self.obs_history[0] if self.obs_history else observation)
                
                # Run inference with the diffusion model
                with torch.no_grad():
                    # Use the correct API for DiffusionPolicy
                    # The from_pretrained model has a different interface
                    if hasattr(self.model, 'predict'):
                        lerobot_actions = self.model.predict(observation)
                    elif hasattr(self.model, 'select_action'):
                        lerobot_actions = self.model.select_action(observation)
                    else:
                        # Fallback to direct forward pass
                        lerobot_actions = self.model(observation)
                    
                # Extract the action for current step
                if isinstance(lerobot_actions, dict) and "action" in lerobot_actions:
                    lerobot_actions = lerobot_actions["action"]
                    
                if isinstance(lerobot_actions, torch.Tensor) and lerobot_actions.dim() > 2:
                    lerobot_actions = lerobot_actions[:, 0, :]  # Take first action step
                
            except Exception as e:
                print(f"[WARNING] Model inference failed: {e}, using fallback")
                lerobot_actions = self._generate_fallback_actions(batch_size)
                
        else:
            # Use placeholder policy with simple walking pattern
            lerobot_actions = self._generate_fallback_actions(batch_size)
        
        # Map the 26 LeRobot actions to 37 Isaac Lab actions
        isaac_actions = self._map_lerobot_to_isaac_actions(lerobot_actions)
        
        self.time_step += 1
        
        return isaac_actions
    
    def _map_isaac_to_lerobot_state(self, isaac_state):
        """Map Isaac Lab's 123D state to LeRobot's 26D state."""
        # Isaac Lab observation structure (123D):
        # - Base linear velocity (3D)
        # - Base angular velocity (3D) 
        # - Projected gravity (3D)
        # - Velocity commands (3D)
        # - Joint positions (37D)
        # - Joint velocities (37D)
        # - Previous actions (37D)
        
        batch_size = isaac_state.shape[0]
        lerobot_state = torch.zeros((batch_size, self.state_dim), device=self.device)
        
        # Extract relevant components from Isaac Lab state
        # This mapping depends on how the original LeRobot model was trained
        # For now, we'll use a simple approach taking the first 26 dimensions
        
        if isaac_state.shape[1] >= 26:
            lerobot_state = isaac_state[:, :26]
        else:
            lerobot_state[:, :isaac_state.shape[1]] = isaac_state
            
        return lerobot_state
    
    def _generate_fallback_actions(self, batch_size):
        """Generate fallback actions when model is not available or fails."""
        lerobot_actions = torch.zeros((batch_size, 26), device=self.device, dtype=torch.float32)
        
        # Simple walking pattern based on time
        t = self.time_step * 0.02  # time in seconds
        
        # Add a simple periodic motion to demonstrate the robot movement
        amplitude = 0.1
        frequency = 1.0
        
        # Leg joints - simple walking pattern
        for i in range(batch_size):
            # Hip pitch joints (index 2, 7 approximately)
            if len(lerobot_actions[i]) > 2:
                lerobot_actions[i, 2] = amplitude * np.sin(frequency * t)  # Left hip
            if len(lerobot_actions[i]) > 7:
                lerobot_actions[i, 7] = amplitude * np.sin(frequency * t + np.pi)  # Right hip
            
            # Knee joints (index 3, 8 approximately)
            if len(lerobot_actions[i]) > 3:
                lerobot_actions[i, 3] = amplitude * np.abs(np.sin(frequency * t))  # Left knee
            if len(lerobot_actions[i]) > 8:
                lerobot_actions[i, 8] = amplitude * np.abs(np.sin(frequency * t + np.pi))  # Right knee
                
            # Ankle joints (index 4, 9 approximately)
            if len(lerobot_actions[i]) > 4:
                lerobot_actions[i, 4] = -amplitude * 0.5 * np.sin(frequency * t)  # Left ankle
            if len(lerobot_actions[i]) > 9:
                lerobot_actions[i, 9] = -amplitude * 0.5 * np.sin(frequency * t + np.pi)  # Right ankle
        
        # Add some small random noise for more natural movement
        lerobot_actions += torch.randn_like(lerobot_actions) * 0.01
        
        return lerobot_actions
    
    def _map_lerobot_to_isaac_actions(self, lerobot_actions):
        """Map 26D LeRobot actions to 37D Isaac Lab actions."""
        batch_size = lerobot_actions.shape[0]
        isaac_actions = torch.zeros((batch_size, 37), device=self.device, dtype=torch.float32)
        
        # Copy the first 26 actions from LeRobot directly
        isaac_actions[:, :26] = lerobot_actions
        
        # Handle finger joints (joints 26-36) - map from 5-finger to 3-finger
        finger_actions = self._map_finger_actions(lerobot_actions)
        isaac_actions[:, 26:] = finger_actions
        
        return isaac_actions

def main():
    """Main function to run the Lerobot policy with G1."""
    
    # Import here to avoid numpy conflicts during Isaac Lab startup
    print("[INFO] Setting up Isaac Lab environment...")
    
    try:
        # Try to import isaaclab_tasks with graceful fallback
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        print("[INFO] Isaac Lab tasks loaded successfully")
    except ImportError as e:
        print(f"[WARNING] Could not import isaaclab_tasks: {e}")
        print("[INFO] Creating basic Isaac Lab environment instead")
        
        # Fallback to basic Isaac Lab setup
        from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg
        from isaaclab.utils import configclass
        
        # Create a minimal environment configuration
        @configclass
        class G1EnvCfg(ManagerBasedRLEnvCfg):
            def __post_init__(self):
                # Scene settings
                self.scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
                # Simulation settings
                self.sim = SimulationCfg(dt=1/60.0)
        
        # Create the environment
        env_cfg = G1EnvCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print("[INFO] Basic Isaac Lab environment created")
        
        # Create dummy observations for the policy
        obs = {
            "policy": torch.zeros((args.num_envs, 123), device="cuda")
        }
        
        # Get environment info
        num_envs = args.num_envs
        env_device = "cuda"
        
        # Simple simulation loop without full gymnasium interface
        print("[INFO] Starting basic simulation...")
        print("[INFO] Press Ctrl+C to stop.")
        
        # Load the Lerobot policy with fake checkpoint for demonstration
        fake_checkpoint_path = "/tmp/fake_checkpoint"
        os.makedirs(fake_checkpoint_path, exist_ok=True)
        
        # Create minimal config for demonstration
        demo_config = {
            "output_features": {"action": {"shape": [26]}},
            "input_features": {"observation.state": {"shape": [26]}},
            "n_obs_steps": 1
        }
        
        with open(os.path.join(fake_checkpoint_path, "config.json"), 'w') as f:
            json.dump(demo_config, f)
        
        policy = LerobotPolicy(fake_checkpoint_path, device=env_device)
        
        timestep = 0
        while simulation_app.is_running():
            start_time = time.time()
            
            # Run policy inference
            with torch.inference_mode():
                actions = policy(obs)
            
            # Update observations (simulate robot state changes)
            obs["policy"] += torch.randn_like(obs["policy"]) * 0.01
            
            timestep += 1
            
            # Print information periodically
            if timestep % 60 == 0:
                print(f"[INFO] Timestep: {timestep}")
                if policy.use_real_model:
                    print(f"[INFO] Robot controlled by real LeRobot diffusion policy")
                else:
                    print(f"[INFO] Robot performing placeholder walking pattern")
            
            # Real-time execution
            if args.real_time:
                elapsed_time = time.time() - start_time
                sleep_time = 1/60.0 - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        print("[INFO] Basic simulation ended.")
        return
    
    # Full Isaac Lab environment setup (if isaaclab_tasks is available)
    # Create the Isaac Lab environment
    task_name = "Isaac-Velocity-Flat-G1-Play-v0"
    print(f"[INFO] Creating environment: {task_name}")
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=args.num_envs)
    
    # Create environment with rendering enabled  
    import gymnasium as gym
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    
    # Get environment info (with type safety)
    num_envs = getattr(env.unwrapped, 'num_envs', args.num_envs)
    env_device = getattr(env.unwrapped, 'device', "cuda")
    
    print(f"[INFO] Environment created with {num_envs} environments")
    
    # Load the Lerobot policy
    policy = LerobotPolicy(args.checkpoint_path, device=env_device)
    
    # Reset environment
    obs, _ = env.reset()
    print(f"[INFO] Initial observation keys: {list(obs.keys())}")
    
    # Get environment step time for real-time execution
    dt = 1.0 / 60.0  # Assume 60 FPS
    step_dt = getattr(env.unwrapped, 'step_dt', dt)
    if step_dt:
        dt = step_dt
    
    print(f"[INFO] Environment step time: {dt:.4f} seconds")
    print(f"[INFO] Starting simulation...")
    print(f"[INFO] The G1 robot should now be visible in Isaac Sim.")
    
    if policy.use_real_model:
        print(f"[INFO] Using LeRobot diffusion model for control.")
    else:
        print(f"[INFO] Using placeholder policy with simple walking pattern.")
        print(f"[INFO] Install LeRobot in 'lerobot' conda environment to use the actual model.")
    
    print(f"[INFO] Press Ctrl+C to stop the simulation.")
    
    timestep = 0
    
    # Main simulation loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run policy inference
        with torch.inference_mode():
            actions = policy(obs)
        
        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        timestep += 1
        
        # Print some information periodically
        if timestep % 60 == 0:
            # Handle rewards safely (could be tensor or scalar)
            try:
                if isinstance(rewards, torch.Tensor):
                    if rewards.numel() > 1:
                        reward_val = rewards.mean().item()
                    else:
                        reward_val = rewards.item()
                else:
                    reward_val = float(rewards)
            except (AttributeError, TypeError, ValueError):
                reward_val = 0.0
                
            print(f"[INFO] Timestep: {timestep}, Rewards: {reward_val:.3f}")
            
            if policy.use_real_model:
                print(f"[INFO] Robot is controlled by LeRobot diffusion policy.")
            else:
                print(f"[INFO] Robot is performing a simple walking pattern (placeholder policy).")
        
        # Handle real-time execution
        if args.real_time:
            elapsed_time = time.time() - start_time
            sleep_time = dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Clean up
    env.close()
    print("[INFO] Simulation ended.")

if __name__ == "__main__":
    main()
    simulation_app.close()
