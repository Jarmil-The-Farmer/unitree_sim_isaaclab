#!/usr/bin/env python3
"""
LeRobot Diffusion Policy Replay with Prerecorded Images
Combines teleoperation_replay.py robot setup with play_lerobot_g1_gui.py diffusion policy loading
Feeds the diffusion model with prerecorded camera images from the dataset
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            return True
    
    print("[WARNING] LeRobot not found in expected locations")
    return False

# Initialize LeRobot availability
LEROBOT_AVAILABLE = ensure_lerobot_available()

from isaaclab.app import AppLauncher

# Add argparse arguments for AppLauncher
parser = argparse.ArgumentParser(description="Run LeRobot diffusion policy with prerecorded images on G1 robot.")
parser.add_argument("--episode_index", type=int, default=0, help="Episode index to load camera images from")
parser.add_argument("--checkpoint_path", type=str, 
                   default="/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot/outputs/train/2025-06-03/00-52-31_diffusion/checkpoints/last/pretrained_model", 
                   help="Path to LeRobot checkpoint directory")
parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor for joint motions")
parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch the Isaac Sim app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


class DiffusionReplayProvider:
    """Provides actions by running diffusion policy on prerecorded camera images."""
    
    def __init__(self, episode_path: str, checkpoint_path: str, scale_factor: float = 1.0, playback_speed: float = 1.0):
        """
        Initialize diffusion replay provider.
        
        Args:
            episode_path: Path to episode data directory containing images and JSON
            checkpoint_path: Path to LeRobot diffusion model checkpoint
            scale_factor: Scale factor for joint motions (default: 1.0)
            playback_speed: Playback speed multiplier (default: 1.0)
        """
        self.episode_path = Path(episode_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.scale_factor = scale_factor
        self.playback_speed = playback_speed
        self.current_frame = 0
        self.time = 0.0
        self.dt = 1.0 / 500.0  # Controller runs at 500Hz
        
        # Load episode metadata
        self._load_episode_metadata()
        
        # Initialize diffusion model
        self._load_diffusion_model()
        
        # Initialize frame timing
        self.frame_duration = 1.0 / 30.0  # 30 FPS original data
        self.next_frame_time = 0.0
        
        print(f"🎬 Diffusion Replay Provider Initialized")
        print(f"   📁 Episode: {self.episode_path.name}")
        print(f"   📊 Total frames: {self.total_frames}")
        print(f"   ⏱️  Duration: {self.total_frames * self.frame_duration:.1f}s")
        print(f"   🎚️  Scale factor: {scale_factor}")
        print(f"   ⚡ Playback speed: {playback_speed}x")
        print(f"   🤖 Using diffusion model: {self.use_real_model}")
        
    def _load_episode_metadata(self):
        """Load episode metadata and check for required files."""
        # Load JSON metadata
        json_path = self.episode_path / "data.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            episode_frames = data.get('data', [])
            self.total_frames = len(episode_frames)
            print(f"✅ Loaded episode metadata: {self.total_frames} frames")
        else:
            # Fallback: count image files
            colors_dir = self.episode_path / "colors"
            if colors_dir.exists():
                image_files = list(colors_dir.glob("*_color_0.jpg"))
                self.total_frames = len(image_files)
                print(f"✅ Counted {self.total_frames} frames from image files")
            else:
                raise ValueError(f"No data.json or colors directory found in {self.episode_path}")
        
        # Check for camera images
        self.colors_dir = self.episode_path / "colors"
        if not self.colors_dir.exists():
            raise ValueError(f"Colors directory not found: {self.colors_dir}")
            
        # Verify first frame images exist
        left_img_path = self.colors_dir / "000000_color_0.jpg"
        right_img_path = self.colors_dir / "000000_color_1.jpg"
        
        if not left_img_path.exists() or not right_img_path.exists():
            raise ValueError(f"Camera images not found: {left_img_path}, {right_img_path}")
            
        print(f"✅ Camera images directory validated: {self.colors_dir}")
        
    def _load_diffusion_model(self):
        """Load LeRobot diffusion model."""
        self.use_real_model = LEROBOT_AVAILABLE
        
        if not self.use_real_model:
            print(f"⚠️  LeRobot not available, using placeholder policy")
            return
            
        try:
            # Load configuration
            config_path = self.checkpoint_path / "config.json"
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            print(f"✅ Loaded diffusion model config from: {config_path}")
            
            # Set up LeRobot imports
            import sys
            lerobot_paths = [
                "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot/lerobot",
                "/home/vaclav/IsaacLab/Unitree G1 - Jarmil the Farmer pretrained data/unitree_IL_lerobot/unitree_lerobot"
            ]
            for path in lerobot_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
            from lerobot.configs.types import FeatureType, NormalizationMode
            from safetensors.torch import load_file
            
            # Create FeatureConfig objects
            class SimpleFeatureConfig:
                def __init__(self, type, shape, **kwargs):
                    if isinstance(type, str):
                        if type == "STATE":
                            self.type = FeatureType.STATE
                        elif type == "VISUAL":
                            self.type = FeatureType.VISUAL
                        elif type == "ACTION":
                            self.type = FeatureType.ACTION
                        else:
                            self.type = FeatureType(type)
                    else:
                        self.type = type
                    
                    self.shape = shape
                    if isinstance(self.shape, tuple):
                        self.shape = list(self.shape)
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            # Process config
            processed_config = self.config.copy()
            
            # Convert input/output features
            if "input_features" in processed_config:
                input_features = {}
                for key, feature_dict in processed_config["input_features"].items():
                    input_features[key] = SimpleFeatureConfig(**feature_dict)
                processed_config["input_features"] = input_features
            
            if "output_features" in processed_config:
                output_features = {}
                for key, feature_dict in processed_config["output_features"].items():
                    output_features[key] = SimpleFeatureConfig(**feature_dict)
                processed_config["output_features"] = output_features
            
            # Convert normalization mapping
            if "normalization_mapping" in processed_config:
                norm_mapping = {}
                for key, norm_str in processed_config["normalization_mapping"].items():
                    if isinstance(norm_str, str):
                        norm_mapping[key] = NormalizationMode(norm_str)
                    else:
                        norm_mapping[key] = norm_str
                processed_config["normalization_mapping"] = norm_mapping
            
            # Create model
            config = DiffusionConfig(**processed_config)
            self.model = DiffusionPolicy(config)
            
            # Load weights
            checkpoint_file = self.checkpoint_path / "diffusion_pytorch_model.safetensors"
            if checkpoint_file.exists():
                state_dict = load_file(str(checkpoint_file))
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.model.to("cuda")
                print(f"✅ Successfully loaded diffusion model from: {checkpoint_file}")
            else:
                print(f"❌ Checkpoint file not found: {checkpoint_file}")
                self.use_real_model = False
                
        except Exception as e:
            print(f"❌ Failed to load diffusion model: {e}")
            import traceback
            traceback.print_exc()
            self.use_real_model = False
            
        # Initialize observation history
        if self.use_real_model:
            self.obs_history = []
            self.n_obs_steps = self.config.get("n_obs_steps", 1)
            
    def _load_camera_images(self, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load left and right camera images for given frame."""
        frame_str = f"{frame_idx:06d}"
        left_img_path = self.colors_dir / f"{frame_str}_color_0.jpg"
        right_img_path = self.colors_dir / f"{frame_str}_color_1.jpg"
        
        # Load images
        try:
            left_img = Image.open(left_img_path).convert('RGB')
            right_img = Image.open(right_img_path).convert('RGB')
            
            # Convert to tensors and normalize
            # Assuming the model expects images in [0, 1] range
            left_tensor = torch.from_numpy(np.array(left_img)).float() / 255.0
            right_tensor = torch.from_numpy(np.array(right_img)).float() / 255.0
            
            # Rearrange to CHW format
            left_tensor = left_tensor.permute(2, 0, 1)  # HWC -> CHW
            right_tensor = right_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Add batch dimension
            left_tensor = left_tensor.unsqueeze(0)  # 1 x C x H x W
            right_tensor = right_tensor.unsqueeze(0)  # 1 x C x H x W
            
            return left_tensor.cuda(), right_tensor.cuda()
            
        except Exception as e:
            print(f"❌ Failed to load images for frame {frame_idx}: {e}")
            # Return dummy images
            dummy_img = torch.zeros((1, 3, 480, 640), device="cuda")
            return dummy_img, dummy_img
    
    def _generate_placeholder_actions(self) -> torch.Tensor:
        """Generate placeholder actions when diffusion model is not available."""
        # Simple walking pattern
        t = self.time
        amplitude = 0.1
        frequency = 1.0
        
        # Create 26D action (based on LeRobot training)
        action = torch.zeros(26, device="cuda")
        
        # Hip pitch joints
        action[2] = amplitude * np.sin(frequency * t)  # Left hip
        action[7] = amplitude * np.sin(frequency * t + np.pi)  # Right hip
        
        # Knee joints
        action[3] = amplitude * np.abs(np.sin(frequency * t))  # Left knee
        action[8] = amplitude * np.abs(np.sin(frequency * t + np.pi))  # Right knee
        
        # Ankle joints
        action[4] = -amplitude * 0.5 * np.sin(frequency * t)  # Left ankle
        action[9] = -amplitude * 0.5 * np.sin(frequency * t + np.pi)  # Right ankle
        
        return action
        
    def _map_to_g1_joints(self, lerobot_actions: torch.Tensor, joint_names: List[str]) -> torch.Tensor:
        """Map LeRobot actions to G1 robot joints with inspire hands."""
        # Create action tensor for all G1 joints
        g1_action = torch.zeros(len(joint_names), dtype=torch.float32, device="cuda")
        
        try:
            # Direct mapping for body joints (first 26 from LeRobot)
            # This assumes the LeRobot model was trained with similar joint ordering
            for i in range(min(26, len(joint_names))):
                g1_action[i] = lerobot_actions[i] * self.scale_factor
            
            # Handle inspire hand joints (if any)
            # This would need to be adapted based on your specific joint mapping
            # For now, we'll leave hand joints at zero (neutral position)
            
            # Find hand joints and set them to neutral
            for i, joint_name in enumerate(joint_names):
                if any(hand_keyword in joint_name.lower() for hand_keyword in ['finger', 'thumb', 'hand']):
                    g1_action[i] = 0.0  # Neutral hand position
                    
        except Exception as e:
            print(f"❌ Error mapping actions to G1 joints: {e}")
            
        return g1_action
        
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from diffusion model using current frame camera images."""
        
        try:
            # Debug first call
            if self.time == 0.0:
                print(f"🔥 Diffusion replay starting!")
                print(f"   Device: {env.device}")
                
                if hasattr(env, 'scene') and 'robot' in env.scene:
                    robot = env.scene['robot']
                    if hasattr(robot, 'data') and hasattr(robot.data, 'joint_names'):
                        joint_names = robot.data.joint_names
                        print(f"   Robot found with {len(joint_names)} joints")
                    else:
                        print(f"   Robot found but no joint_names accessible")
                else:
                    print(f"❌ No robot found in scene!")
                    return None
            
            # Check if we should advance to next frame
            if self.time >= self.next_frame_time:
                if self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                    self.next_frame_time += self.frame_duration / self.playback_speed
                else:
                    if int(self.time * 500) % 250 == 0:
                        print(f"🏁 Episode complete - holding at final frame {self.current_frame}/{self.total_frames}")
            
            # Check if replay is completed
            if self.current_frame >= self.total_frames:
                print(f"🏁 Replay completed! {self.total_frames} frames processed")
                return None
            
            # Get actions from diffusion model
            if self.use_real_model:
                # Load camera images for current frame
                left_img, right_img = self._load_camera_images(self.current_frame)
                
                # Create observation dictionary for the model
                observation = {
                    "observation.images.cam_left_high": left_img,
                    "observation.images.cam_right_high": right_img,
                }
                
                # Add state observation if needed (you might need to create dummy state)
                if "observation.state" in self.config.get("input_features", {}):
                    state_dim = self.config["input_features"]["observation.state"]["shape"][0]
                    dummy_state = torch.zeros((1, state_dim), device="cuda")
                    observation["observation.state"] = dummy_state
                
                # Run diffusion model inference
                with torch.no_grad():
                    if hasattr(self.model, 'predict'):
                        model_output = self.model.predict(observation)
                    elif hasattr(self.model, 'select_action'):
                        model_output = self.model.select_action(observation)
                    else:
                        model_output = self.model(observation)
                    
                    # Extract action
                    if isinstance(model_output, dict) and "action" in model_output:
                        lerobot_actions = model_output["action"]
                    else:
                        lerobot_actions = model_output
                    
                    # Handle temporal dimension
                    if lerobot_actions.dim() > 2:
                        lerobot_actions = lerobot_actions[:, 0, :]  # Take first action step
                    
                    lerobot_actions = lerobot_actions.squeeze(0)  # Remove batch dimension
                    
            else:
                # Use placeholder policy
                lerobot_actions = self._generate_placeholder_actions()
            
            # Map to G1 robot joints
            if hasattr(env, 'scene') and 'robot' in env.scene:
                robot = env.scene['robot']
                if hasattr(robot, 'data') and hasattr(robot.data, 'joint_names'):
                    joint_names = robot.data.joint_names
                    action = self._map_to_g1_joints(lerobot_actions, joint_names)
                    
                    # Add batch dimension and move to correct device
                    action = action.unsqueeze(0).to(env.device)
                    
                    # Update time
                    self.time += self.dt
                    
                    # Progress output
                    if int(self.time * 500) % 500 == 0:
                        progress_percent = 100 * self.current_frame / self.total_frames
                        remaining_frames = self.total_frames - self.current_frame
                        estimated_time_left = remaining_frames * self.frame_duration / self.playback_speed
                        print(f"🎬 Frame {self.current_frame:4d}/{self.total_frames} | {progress_percent:5.1f}% | ⏱️ {self.time:.1f}s | ETA: {estimated_time_left:.1f}s")
                    
                    return action
                else:
                    print(f"❌ Robot found but no joint_names accessible!")
                    return None
            else:
                print(f"❌ No robot found in scene!")
                return None
                
        except Exception as e:
            print(f"❌ Exception in get_action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset(self):
        """Reset the replay to beginning."""
        self.current_frame = 0
        self.time = 0.0
        self.next_frame_time = 0.0
        print("🔄 Diffusion replay reset")


def main():
    """Main function to run diffusion policy replay."""
    
    # Create episode path
    episode_path = f"/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data/episode_{args.episode_index:04d}"
    
    if not os.path.exists(episode_path):
        available_episodes = []
        data_dir = "/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data"
        if os.path.exists(data_dir):
            available_episodes = [d for d in os.listdir(data_dir) if d.startswith('episode_')]
        print(f"❌ Episode {args.episode_index} not found at {episode_path}")
        print(f"Available episodes: {len(available_episodes)}")
        return
    
    print(f"[INFO] Setting up Isaac Lab environment...")
    
    # Run with the sim_main.py infrastructure
    print(f"[INFO] Starting diffusion policy replay...")
    print(f"   Episode: {args.episode_index}")
    print(f"   Checkpoint: {args.checkpoint_path}")
    print(f"   Scale factor: {args.scale_factor}")
    print(f"   Playback speed: {args.playback_speed}x")
    
    # Note: This script is designed to work with the existing sim_main.py infrastructure
    # Create the provider for integration
    provider = DiffusionReplayProvider(
        episode_path=episode_path,
        checkpoint_path=args.checkpoint_path,
        scale_factor=args.scale_factor,
        playback_speed=args.playback_speed
    )
    
    print(f"✅ Diffusion replay provider created successfully!")
    print(f"[INFO] To run with Isaac Sim, use:")
    print(f"python unitree_sim_isaaclab/sim_main.py --task Isaac-Simple-G1-Inspire --action_source diffusion --episode_index {args.episode_index} --scale_factor {args.scale_factor} --playback_speed {args.playback_speed}")


if __name__ == "__main__":
    main()
    simulation_app.close()
