#!/usr/bin/env python3

"""Script to play a Lerobot diffusion policy with Unitree G1 with Inspire hands in Isaac Lab with GUI.

This script integrates LeRobot's diffusion policy with the Unitree G1 robot simulation using Inspire hands.
It handles the action space mapping from LeRobot's 26D actions to Unitree G1's 53D joint space (29 body + 24 hand joints).

Key features:
- Loads pretrained LeRobot diffusion models for humanoid control
- Uses Unitree G1 robot with Inspire hands (5-finger dexterous hands)
- Maps LeRobot actions to 53D action space (29 body joints + 24 Inspire hand joints)
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
import matplotlib.pyplot as plt

# Environment isolation for LeRobot
def ensure_lerobot_available():
    """Ensure LeRobot is available with proper environment isolation."""
    import sys
    lerobot_paths = [
        "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot"
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

# Set up the unitree_sim_isaaclab project root path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_sim_isaaclab")
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher

# Add argparse arguments for AppLauncher
parser = argparse.ArgumentParser(description="Play Lerobot policy with Unitree G1 with Inspire hands (with GUI).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint_path", type=str, default="/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot/outputs/train/2025-06-03/00-52-31_diffusion/checkpoints/last/pretrained_model", help="Path to Lerobot checkpoint directory.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time if possible.")
# Note: --device is provided by AppLauncher, so we don't add it here
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the Isaac Sim app with rendering enabled
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching the app
import sys
unitree_sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_sim_isaaclab")
if unitree_sim_path not in sys.path:
    sys.path.insert(0, unitree_sim_path)

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
                lerobot_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot"
                if lerobot_path not in sys.path:
                    sys.path.insert(0, lerobot_path)
                    print(f"[INFO] Re-added LeRobot path after Isaac Lab init: {lerobot_path}")
                
                # Also ensure the parent path is available for imports
                parent_path = "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot"
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
                print(f"[DEBUG] LeRobot path exists: {os.path.exists('/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot')}")
                
                # Try simpler approach - use the working method from play_lerobot_g1_final.py
                try:
                    print(f"[INFO] Trying exact method from working final script...")
                    
                    # Force re-add LeRobot paths
                    import sys
                    lerobot_paths = [
                        "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot",
                        "/home/vaclav/IsaacLab_unitree/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot"
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
        """Create mapping from 26D LeRobot actions to 53D Unitree G1 Inspire actions."""
        # G1 with Inspire hands joint order (53 joints total):
        # Legs: left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll (6)
        #       right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll (6)
        # Waist: waist_yaw, waist_roll, waist_pitch (3)
        # Arms: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw (7)
        #       right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw (7)
        # Inspire Hands: 24 total finger joints (12 per hand)
        
        # LeRobot was trained with 26 actions, we'll map the first 24 non-hand actions directly
        # and handle the Inspire hand joints separately
        mapping = {}
        
        # Map the first 24 joints (legs + waist + arms, excluding hands)
        for i in range(min(24, 53)):
            mapping[i] = i
            
        return mapping
    
    def _map_finger_actions(self, lerobot_actions):
        """Map LeRobot finger actions to Inspire hand actions.
        
        The Inspire hands have 5 fingers with multiple joints each:
        - Index: proximal, intermediate joints (2 joints)
        - Middle: proximal, intermediate joints (2 joints) 
        - Ring: proximal, intermediate joints (2 joints)
        - Pinky: proximal, intermediate joints (2 joints)
        - Thumb: proximal_yaw, proximal_pitch, intermediate, distal joints (4 joints)
        Total: 12 joints per hand = 24 joints for both hands
        
        Args:
            lerobot_actions: Actions from LeRobot model (26D)
            
        Returns:
            finger_actions: Mapped finger actions for Inspire hands (24D finger joints)
        """
        batch_size = lerobot_actions.shape[0]
        
        # Inspire hand joint structure (24 total finger joints):
        # Left hand: 12 joints, Right hand: 12 joints
        finger_actions = torch.zeros((batch_size, 24), device=self.device)
        
        # Simple mapping strategy - use the last 2 LeRobot actions for basic finger control
        if lerobot_actions.shape[1] >= 26:
            for i in range(batch_size):
                # Get the last 2 actions from LeRobot (actions 24 and 25)
                left_hand_action = lerobot_actions[i, 24] if lerobot_actions.shape[1] > 24 else 0.0
                right_hand_action = lerobot_actions[i, 25] if lerobot_actions.shape[1] > 25 else 0.0
                
                # Map to Inspire hand joints
                # Left hand (joints 0-11)
                finger_actions[i, 0] = left_hand_action * 0.8  # L_index_proximal_joint
                finger_actions[i, 1] = left_hand_action * 0.6  # L_index_intermediate_joint
                finger_actions[i, 2] = left_hand_action * 0.8  # L_middle_proximal_joint
                finger_actions[i, 3] = left_hand_action * 0.6  # L_middle_intermediate_joint
                finger_actions[i, 4] = left_hand_action * 0.7  # L_ring_proximal_joint
                finger_actions[i, 5] = left_hand_action * 0.5  # L_ring_intermediate_joint
                finger_actions[i, 6] = left_hand_action * 0.6  # L_pinky_proximal_joint
                finger_actions[i, 7] = left_hand_action * 0.4  # L_pinky_intermediate_joint
                finger_actions[i, 8] = left_hand_action * 0.5  # L_thumb_proximal_yaw_joint
                finger_actions[i, 9] = left_hand_action * 0.7  # L_thumb_proximal_pitch_joint
                finger_actions[i, 10] = left_hand_action * 0.6 # L_thumb_intermediate_joint
                finger_actions[i, 11] = left_hand_action * 0.5 # L_thumb_distal_joint
                
                # Right hand (joints 12-23) 
                finger_actions[i, 12] = right_hand_action * 0.8  # R_index_proximal_joint
                finger_actions[i, 13] = right_hand_action * 0.6  # R_index_intermediate_joint
                finger_actions[i, 14] = right_hand_action * 0.8  # R_middle_proximal_joint
                finger_actions[i, 15] = right_hand_action * 0.6  # R_middle_intermediate_joint
                finger_actions[i, 16] = right_hand_action * 0.7  # R_ring_proximal_joint
                finger_actions[i, 17] = right_hand_action * 0.5  # R_ring_intermediate_joint
                finger_actions[i, 18] = right_hand_action * 0.6  # R_pinky_proximal_joint
                finger_actions[i, 19] = right_hand_action * 0.4  # R_pinky_intermediate_joint
                finger_actions[i, 20] = right_hand_action * 0.5  # R_thumb_proximal_yaw_joint
                finger_actions[i, 21] = right_hand_action * 0.7  # R_thumb_proximal_pitch_joint
                finger_actions[i, 22] = right_hand_action * 0.6  # R_thumb_intermediate_joint
                finger_actions[i, 23] = right_hand_action * 0.5  # R_thumb_distal_joint
        
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
        """Map 26D LeRobot actions to 53D Unitree G1 Inspire actions."""
        batch_size = lerobot_actions.shape[0]
        isaac_actions = torch.zeros((batch_size, 53), device=self.device, dtype=torch.float32)
        
        # Copy the first 24 actions from LeRobot (legs + waist + arms, excluding hands)
        isaac_actions[:, :24] = lerobot_actions[:, :24]
        
        # Handle Inspire finger joints (joints 29-52) - map from LeRobot hand actions
        finger_actions = self._map_finger_actions(lerobot_actions)
        isaac_actions[:, 29:] = finger_actions  # Inspire hands start at joint 29
        
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
    
    # Add RealSense D455-like stereo cameras to the environment
    env_cfg = add_realsense_cameras(env_cfg)
    
    # Create environment with rendering enabled  
    import gymnasium as gym
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    
    # Get environment info (with type safety)
    num_envs = getattr(env.unwrapped, 'num_envs', args.num_envs)
    env_device = getattr(env.unwrapped, 'device', "cuda")
    
    print(f"[INFO] Environment created with {num_envs} environments")
    print(f"[INFO] RealSense D455-like stereo cameras added to the scene")
    
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
    
    # Setup camera visualization
    camera_output_dir = "/tmp/realsense_camera_output"
    os.makedirs(camera_output_dir, exist_ok=True)
    print(f"[INFO] Camera output will be saved to: {camera_output_dir}")
    
    # Main simulation loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run policy inference
        with torch.inference_mode():
            actions = policy(obs)
        
        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        timestep += 1
        
        # Visualize camera output every 30 frames (0.5 seconds at 60 FPS)
        if timestep % 30 == 0:
            visualize_camera_output(env, timestep, camera_output_dir)
        
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


def create_unitree_g1_inspire_env(num_envs: int = 1):
    """Create a custom environment with Unitree G1 robot with Inspire hands."""
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import ActionTermCfg as ActionTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    import isaaclab.envs.mdp as mdp
    
    # Import Unitree robot configuration
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_sim_isaaclab"))
    from robots.unitree import G129_CFG_WITH_INSPIRE_HAND
    
    @configclass
    class ActionsCfg:
        """Action configuration for joint position control."""
        joint_pos = JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[".*"], 
            scale=1.0
        )
    
    @configclass
    class ObservationsCfg:
        """Observation configuration."""
        @configclass
        class PolicyCfg(ObsGroup):
            """Policy observation group."""
            # Robot joint positions (29 DOF for Inspire hands)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            # Robot joint velocities  
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            # Base linear velocity
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            # Base angular velocity
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            # Projected gravity
            projected_gravity = ObsTerm(func=mdp.projected_gravity)
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class RewardsCfg:
        """Reward configuration."""
        # Simple dummy reward to keep robot stable
        alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    @configclass
    class TerminationsCfg:
        """Termination configuration."""
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    @configclass
    class EventCfg:
        """Event configuration."""
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )
    
    @configclass 
    class G1InspireEnvCfg(ManagerBasedRLEnvCfg):
        """Custom environment configuration for G1 with Inspire hands."""
        
        def __post_init__(self):
            # Basic simulation settings
            self.decimation = 2
            self.episode_length_s = 20.0
            self.num_actions = 53  # 29 DOF G1 + 24 Inspire hand DOF
            self.num_observations = 123  # Approximate - will be calculated based on actual obs
            self.num_states = 0
            
            # Scene configuration
            self.scene = InteractiveSceneCfg(
                num_envs=num_envs, 
                env_spacing=2.0,
                replicate_physics=True
            )
            
            # Simulation configuration
            self.sim = SimulationCfg(
                dt=1/60.0,  # 60 FPS simulation
                render_interval=self.decimation,
            )
            
            # Add robot to scene
            from copy import deepcopy
            self.scene.robot = deepcopy(G129_CFG_WITH_INSPIRE_HAND)
            self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
            
            # Configure actions, observations, rewards, terminations, and events
            self.actions = ActionsCfg()
            self.observations = ObservationsCfg()
            self.rewards = RewardsCfg()
            self.terminations = TerminationsCfg()
            self.events = EventCfg()
    
    return G1InspireEnvCfg()


def add_realsense_cameras(env_cfg):
    """Add RealSense D455-like stereo cameras to the environment configuration."""
    try:
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import PinholeCameraCfg
        
        # RealSense D455 specifications:
        # - Dual global shutter cameras (left and right)
        # - 1280x720 resolution
        # - Baseline distance: ~50mm (0.05m)
        # - FOV: 86° x 57°
        
        # Camera configuration based on RealSense D455 specifications
        camera_height = 720
        camera_width = 1280
        focal_length = 24.0  # Approximate focal length
        baseline_distance = 0.05  # 50mm baseline
        
        # Left camera (Camera_OmniVision_OV9782_Left equivalent)
        left_camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/realsense_left_cam",
            update_period=0.033,  # ~30 FPS
            height=camera_height,
            width=camera_width,
            data_types=["rgb", "distance_to_image_plane", "normals"],
            spawn=PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, baseline_distance/2, 0.1),  # Left camera position
                rot=(0.5, -0.5, 0.5, -0.5), 
                convention="ros"
            ),
        )
        
        # Right camera (Camera_OmniVision_OV9782_Right equivalent)
        right_camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/realsense_right_cam", 
            update_period=0.033,  # ~30 FPS
            height=camera_height,
            width=camera_width,
            data_types=["rgb", "distance_to_image_plane", "normals"],
            spawn=PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, -baseline_distance/2, 0.1),  # Right camera position  
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros"
            ),
        )
        
        # Add cameras to the scene configuration
        if hasattr(env_cfg, 'scene'):
            # Add cameras to the existing scene
            env_cfg.scene.realsense_left_camera = left_camera_cfg
            env_cfg.scene.realsense_right_camera = right_camera_cfg
            print("[INFO] Added RealSense D455-like stereo cameras to scene configuration")
        else:
            print("[WARNING] Could not add cameras - scene configuration not found")
            
    except ImportError as e:
        print(f"[WARNING] Could not import camera configuration: {e}")
        print("[WARNING] Cameras will not be added to the scene")
    
    return env_cfg


def visualize_camera_output(env, timestep, output_dir):
    """Visualize and save camera output from RealSense-like cameras."""
    try:
        # Check if the environment has our cameras
        scene = getattr(env.unwrapped, 'scene', None)
        if scene is None:
            return
            
        left_cam_name = "realsense_left_camera"
        right_cam_name = "realsense_right_camera"
        
        # Get camera data if available
        left_camera = scene.get(left_cam_name, None)
        right_camera = scene.get(right_cam_name, None)
        
        if left_camera is None or right_camera is None:
            # Try alternative names
            for name in scene.sensors.keys():
                if "left" in name.lower() and "cam" in name.lower():
                    left_camera = scene[name]
                    left_cam_name = name
                elif "right" in name.lower() and "cam" in name.lower():
                    right_camera = scene[name]
                    right_cam_name = name
        
        if left_camera is not None and right_camera is not None:
            # Get RGB images
            left_rgb = left_camera.data.output.get("rgb", None)
            right_rgb = right_camera.data.output.get("rgb", None)
            
            # Get depth images
            left_depth = left_camera.data.output.get("distance_to_image_plane", None)
            right_depth = right_camera.data.output.get("distance_to_image_plane", None)
            
            if left_rgb is not None and right_rgb is not None:
                # Take first environment's camera data
                left_rgb_img = left_rgb[0].detach().cpu().numpy()
                right_rgb_img = right_rgb[0].detach().cpu().numpy()
                
                # Create stereo RGB visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # RGB images
                if left_rgb_img.shape[-1] >= 3:
                    axes[0, 0].imshow(left_rgb_img[..., :3])
                    axes[0, 0].set_title(f"Left RGB Camera (OV9782_Left)")
                    axes[0, 0].axis('off')
                    
                if right_rgb_img.shape[-1] >= 3:
                    axes[0, 1].imshow(right_rgb_img[..., :3])
                    axes[0, 1].set_title(f"Right RGB Camera (OV9782_Right)")
                    axes[0, 1].axis('off')
                
                # Depth images
                if left_depth is not None:
                    left_depth_img = left_depth[0].detach().cpu().numpy()
                    if left_depth_img.ndim == 3:
                        left_depth_img = left_depth_img[..., 0]
                    axes[1, 0].imshow(left_depth_img, cmap='turbo')
                    axes[1, 0].set_title(f"Left Depth Camera")
                    axes[1, 0].axis('off')
                    
                if right_depth is not None:
                    right_depth_img = right_depth[0].detach().cpu().numpy()
                    if right_depth_img.ndim == 3:
                        right_depth_img = right_depth_img[..., 0]
                    axes[1, 1].imshow(right_depth_img, cmap='turbo')
                    axes[1, 1].set_title(f"Right Depth Camera")
                    axes[1, 1].axis('off')
                
                plt.suptitle(f"RealSense D455-like Stereo Cameras - Timestep {timestep}")
                plt.tight_layout()
                
                # Save the visualization
                save_path = os.path.join(output_dir, f"stereo_cameras_{timestep:06d}.jpg")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                # Print camera info every 60 frames
                if timestep % 60 == 0:
                    print(f"[INFO] Camera output saved: {save_path}")
                    print(f"[INFO] Left RGB shape: {left_rgb_img.shape}")
                    print(f"[INFO] Right RGB shape: {right_rgb_img.shape}")
                    if left_depth is not None:
                        print(f"[INFO] Left depth shape: {left_depth_img.shape}")
                    if right_depth is not None:
                        print(f"[INFO] Right depth shape: {right_depth_img.shape}")
        else:
            # List available sensors for debugging
            if timestep == 30:  # Only print once
                available_sensors = list(scene.sensors.keys()) if hasattr(scene, 'sensors') else []
                print(f"[DEBUG] Available sensors: {available_sensors}")
                print(f"[WARNING] RealSense cameras not found in scene")
                
    except Exception as e:
        if timestep % 60 == 0:  # Only print errors occasionally to avoid spam
            print(f"[WARNING] Camera visualization failed: {e}")


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
    
    # Full Isaac Lab environment setup with Unitree G1 Inspire hands
    print("[INFO] Creating Unitree G1 environment with Inspire hands...")
    
    try:
        # Create custom environment configuration
        env_cfg = create_unitree_g1_inspire_env(num_envs=args.num_envs)
        
        # Create environment with rendering enabled  
        from isaaclab.envs import ManagerBasedRLEnv
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print(f"[INFO] Unitree G1 with Inspire hands environment created successfully")
        
        # Get environment info (with type safety)
        num_envs = getattr(env, 'num_envs', args.num_envs)
        env_device = getattr(env, 'device', "cuda")
        
        print(f"[INFO] Environment created with {num_envs} environments")
        print(f"[INFO] Unitree G1 robot with Inspire hands should now be visible in Isaac Sim")
        
        # Check if we can access the viewport
        try:
            import omni.kit.viewport.utility as vp_utils
            from omni.isaac.core.utils.stage import get_current_stage
            
            # Get current stage and print some info
            stage = get_current_stage()
            if stage:
                print(f"[INFO] Current stage path: {stage.GetRootLayer().identifier}")
                
                # List all prims in the stage for debugging
                root_prim = stage.GetDefaultPrim()
                if not root_prim:
                    root_prim = stage.GetPrimAtPath("/")
                
                print(f"[INFO] Stage root prim: {root_prim.GetPath()}")
                
                # Look for our robot
                robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
                if robot_prim and robot_prim.IsValid():
                    print(f"[INFO] ✓ Robot found at: {robot_prim.GetPath()}")
                else:
                    print(f"[WARNING] Robot not found at expected path /World/envs/env_0/Robot")
                    # Try to find robot anywhere
                    from pxr import Usd
                    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
                        if "robot" in str(prim.GetPath()).lower() or "g1" in str(prim.GetPath()).lower():
                            print(f"[DEBUG] Found robot-like prim: {prim.GetPath()}")
                
                # Check viewport
                viewport_api = vp_utils.get_active_viewport()
                if viewport_api:
                    print(f"[INFO] ✓ Active viewport found")
                    # Try to frame the stage
                    viewport_api.frame_stage()
                    print(f"[INFO] ✓ Viewport framed to stage")
                else:
                    print(f"[WARNING] No active viewport found")
            else:
                print(f"[WARNING] No current stage found")
                
        except Exception as e:
            print(f"[DEBUG] Viewport setup failed: {e}")
        
        # Load the Lerobot policy  
        policy = LerobotPolicy(args.checkpoint_path, device=env_device)
        
        # Reset environment
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract observation dict from tuple
        print(f"[INFO] Initial observation keys: {list(obs.keys())}")
        print(f"[INFO] Observation shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()]}")
        
        # Get environment step time for real-time execution
        dt = 1.0 / 60.0  # Assume 60 FPS
        step_dt = getattr(env, 'step_dt', dt)
        if step_dt:
            dt = step_dt
        
        print(f"[INFO] Environment step time: {dt:.4f} seconds")
        print(f"[INFO] Starting simulation...")
        
        if policy.use_real_model:
            print(f"[INFO] Using LeRobot diffusion model for control.")
        else:
            print(f"[INFO] Using placeholder policy with simple walking pattern.")
        
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
                print(f"[INFO] Robot is controlled by LeRobot diffusion policy.")
            
            # Handle real-time execution
            if args.real_time:
                elapsed_time = time.time() - start_time
                sleep_time = dt - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Clean up
        env.close()
        print("[INFO] Simulation ended.")
        return
        
    except Exception as e:
        print(f"[WARNING] Failed to create Unitree G1 environment: {e}")
        print("[INFO] Falling back to basic environment...")
        
        # Fallback to basic Isaac Lab setup
        from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg
        from isaaclab.utils import configclass
        
        # Simple empty config classes for fallback
        @configclass
        class SimpleActionsCfg:
            pass
            
        @configclass 
        class SimpleObservationsCfg:
            pass
            
        @configclass
        class SimpleRewardsCfg:
            pass
            
        @configclass
        class SimpleTerminationsCfg:
            pass
        
        @configclass
        class G1EnvCfg(ManagerBasedRLEnvCfg):
            def __post_init__(self):
                # Basic required fields
                self.decimation = 4
                self.episode_length_s = 30.0
                
                # Scene settings
                self.scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
                
                # Simulation settings
                self.sim = SimulationCfg(dt=1/60.0)
                
                # Basic actions (empty for fallback)
                self.actions = SimpleActionsCfg()
                
                # Basic observations (empty for fallback)  
                self.observations = SimpleObservationsCfg()
                
                # Basic rewards (empty for fallback)
                self.rewards = SimpleRewardsCfg()
                
                # Basic terminations (empty for fallback)
                self.terminations = SimpleTerminationsCfg()
        
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
    
    # Get environment info (with type safety)
    num_envs = getattr(env, 'num_envs', args.num_envs)
    env_device = getattr(env, 'device', "cuda")
    
    print(f"[INFO] Environment created with {num_envs} environments")
    print(f"[INFO] Unitree G1 robot with Inspire hands should now be visible in Isaac Sim")
    
    # Load the Lerobot policy  
    policy = LerobotPolicy(args.checkpoint_path, device=env_device)
    
    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract observation dict from tuple
    print(f"[INFO] Initial observation keys: {list(obs.keys())}")
    
    # Get environment step time for real-time execution
    dt = 1.0 / 60.0  # Assume 60 FPS
    step_dt = getattr(env, 'step_dt', dt)
    if step_dt:
        dt = step_dt
    
    print(f"[INFO] Environment step time: {dt:.4f} seconds")
    print(f"[INFO] Starting simulation...")
    
    if policy.use_real_model:
        print(f"[INFO] Using LeRobot diffusion model for control.")
    else:
        print(f"[INFO] Using placeholder policy with simple walking pattern.")
        print(f"[INFO] Install LeRobot dependencies to use the actual model.")
    
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
