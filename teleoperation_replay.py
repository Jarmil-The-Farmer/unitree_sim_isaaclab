#!/usr/bin/env python3
"""
Teleoperation Data Replay Script for Isaac Lab
Converts teleoperation training data to G1 robot with inspire hands format and replays in simulation
"""

import os
import json
import numpy as np
import torch
import sys
import math
from typing import Dict, List, Optional, Tuple

# Add paths for Isaac Lab infrastructure
sys.path.append("/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab/action_provider")
sys.path.append("/home/vaclav/IsaacLab_unitree/IsaacLab/unitree_sim_isaaclab")

from action_base import ActionProvider


class TeleoperationReplayActionProvider(ActionProvider):
    """Action provider that replays teleoperation training data with proper joint mapping."""
    
    def __init__(self, episode_path: str, scale_factor: float = 1.0, playback_speed: float = 1.0):
        """
        Initialize teleoperation replay.
        
        Args:
            episode_path: Path to episode data JSON file
            scale_factor: Scale factor for joint motions (default: 1.0)
            playback_speed: Playback speed multiplier (default: 1.0)
        """
        super().__init__("TeleoperationReplayActionProvider")
        
        self.episode_path = episode_path
        self.scale_factor = scale_factor
        self.playback_speed = playback_speed
        self.current_frame = 0
        self.time = 0.0
        self.dt = 1.0 / 500.0  # Controller runs at 500Hz
        
        # Load episode data
        self.episode_data = self._load_episode_data()
        if not self.episode_data:
            raise ValueError(f"Failed to load episode data from {episode_path}")
        
        self.total_frames = len(self.episode_data)
        self.frame_duration = 1.0 / 30.0  # 30 FPS original data
        self.next_frame_time = 0.0
        
        # Define joint mappings
        self._setup_joint_mappings()
        
        print(f"🎬 Teleoperation Replay Provider Initialized")
        print(f"   📁 Episode: {os.path.basename(episode_path)}")
        print(f"   📊 Total frames: {self.total_frames}")
        print(f"   ⏱️  Duration: {self.total_frames * self.frame_duration:.1f}s")
        print(f"   🎚️  Scale factor: {scale_factor}")
        print(f"   ⚡ Playback speed: {playback_speed}x")
        
    def _load_episode_data(self) -> List[Dict]:
        """Load episode data from JSON file."""
        try:
            with open(self.episode_path, 'r') as f:
                data = json.load(f)
            
            episode_frames = data.get('data', [])
            print(f"✅ Loaded {len(episode_frames)} frames from episode")
            return episode_frames
            
        except Exception as e:
            print(f"❌ Error loading episode data: {e}")
            return []
    
    def _setup_joint_mappings(self):
        """Setup joint name mappings between teleoperation data and G1 robot."""
        
        # Arm joint mapping (7 DOF each)
        self.arm_joint_mapping = {
            # Left arm
            'left_shoulder_pitch_joint': 'left_arm_0',
            'left_shoulder_roll_joint': 'left_arm_1', 
            'left_shoulder_yaw_joint': 'left_arm_2',
            'left_elbow_joint': 'left_arm_3',
            'left_wrist_roll_joint': 'left_arm_4',
            'left_wrist_pitch_joint': 'left_arm_5',
            'left_wrist_yaw_joint': 'left_arm_6',
            
            # Right arm
            'right_shoulder_pitch_joint': 'right_arm_0',
            'right_shoulder_roll_joint': 'right_arm_1',
            'right_shoulder_yaw_joint': 'right_arm_2', 
            'right_elbow_joint': 'right_arm_3',
            'right_wrist_roll_joint': 'right_arm_4',
            'right_wrist_pitch_joint': 'right_arm_5',
            'right_wrist_yaw_joint': 'right_arm_6'
        }
        
        # Hand mapping: 6 recorded parameters -> 12 inspire hand joints
        # Based on user specification: proximal, intermediate, thumb_yaw, thumb_pitch, thumb_intermediate, thumb_distal
        self.hand_joint_mapping = {
            # Left hand inspire joints (12 total)
            'L_index_proximal_joint': 'left_hand_0',      # Index finger proximal
            'L_index_intermediate_joint': 'left_hand_1',  # Index finger intermediate  
            'L_middle_proximal_joint': 'left_hand_0',     # Middle finger proximal (same as index)
            'L_middle_intermediate_joint': 'left_hand_1', # Middle finger intermediate (same as index)
            'L_pinky_proximal_joint': 'left_hand_0',      # Pinky finger proximal (same as index)
            'L_pinky_intermediate_joint': 'left_hand_1',  # Pinky finger intermediate (same as index)
            'L_ring_proximal_joint': 'left_hand_0',       # Ring finger proximal (same as index)
            'L_ring_intermediate_joint': 'left_hand_1',   # Ring finger intermediate (same as index)
            'L_thumb_proximal_yaw_joint': 'left_hand_2',  # Thumb yaw
            'L_thumb_proximal_pitch_joint': 'left_hand_3',# Thumb pitch
            'L_thumb_intermediate_joint': 'left_hand_4',  # Thumb intermediate
            'L_thumb_distal_joint': 'left_hand_5',        # Thumb distal
            
            # Right hand inspire joints (12 total)
            'R_index_proximal_joint': 'right_hand_0',
            'R_index_intermediate_joint': 'right_hand_1',
            'R_middle_proximal_joint': 'right_hand_0',
            'R_middle_intermediate_joint': 'right_hand_1', 
            'R_pinky_proximal_joint': 'right_hand_0',
            'R_pinky_intermediate_joint': 'right_hand_1',
            'R_ring_proximal_joint': 'right_hand_0',
            'R_ring_intermediate_joint': 'right_hand_1',
            'R_thumb_proximal_yaw_joint': 'right_hand_2',
            'R_thumb_proximal_pitch_joint': 'right_hand_3',
            'R_thumb_intermediate_joint': 'right_hand_4',
            'R_thumb_distal_joint': 'right_hand_5'
        }
        
        print(f"🔧 Joint mappings configured:")
        print(f"   🦾 Arms: {len([k for k in self.arm_joint_mapping.keys() if 'left' in k])} left + {len([k for k in self.arm_joint_mapping.keys() if 'right' in k])} right = 14 DOF")
        print(f"   ✋ Hands: {len([k for k in self.hand_joint_mapping.keys() if 'L_' in k])} left + {len([k for k in self.hand_joint_mapping.keys() if 'R_' in k])} right = 24 DOF")
    
    def _convert_hand_servo_to_radians(self, servo_values: List[float]) -> List[float]:
        """
        Convert servo values (600-950 range) to radian values for inspire hand joints.
        
        Args:
            servo_values: List of 6 servo values from teleoperation data
            
        Returns:
            List of 6 radian values
        """
        # Ensure we have exactly 6 values
        if len(servo_values) < 6:
            if self.time == 0.0:  # Only warn on first frame
                print(f"⚠️  Warning: Only {len(servo_values)} servo values, padding to 6")
            servo_values = list(servo_values) + [700.0] * (6 - len(servo_values))
        elif len(servo_values) > 6:
            if self.time == 0.0:  # Only warn on first frame
                print(f"⚠️  Warning: {len(servo_values)} servo values, truncating to 6")
            servo_values = servo_values[:6]
        
        # Convert servo range (600-950) to normalized range (0-1)
        normalized = [(val - 600.0) / 350.0 for val in servo_values]
        
        # Convert to reasonable joint ranges in radians
        # Assuming finger joints can move in range [-0.5, 1.5] radians
        joint_ranges = [
            (-0.5, 1.5),  # proximal joint
            (-0.5, 1.5),  # intermediate joint  
            (-1.0, 1.0),  # thumb yaw
            (-1.0, 1.0),  # thumb pitch
            (-0.5, 1.5),  # thumb intermediate
            (-0.5, 1.5)   # thumb distal
        ]
        
        radians = []
        for i, (norm_val, (min_rad, max_rad)) in enumerate(zip(normalized, joint_ranges)):
            # Clamp normalized value
            norm_val = max(0.0, min(1.0, norm_val))
            # Convert to radian range
            rad_val = min_rad + norm_val * (max_rad - min_rad)
            radians.append(rad_val)
        
        return radians
    
    def _get_current_frame_actions(self) -> Optional[Dict]:
        """Get actions for current frame."""
        # Check bounds first
        if self.current_frame >= self.total_frames:
            print(f"⚠️  Frame {self.current_frame} is out of bounds (total: {self.total_frames})")
            return None
        
        # Use 0-based indexing but ensure we don't go negative    
        frame_index = max(0, min(self.current_frame, self.total_frames - 1))
        
        if frame_index != self.current_frame:
            print(f"⚠️  Adjusted frame {self.current_frame} to {frame_index}")
            
        frame_data = self.episode_data[frame_index]
        actions = frame_data.get('actions', {})
        
        return actions
    
    def _map_to_robot_joints(self, actions: Dict, joint_names: List[str]) -> torch.Tensor:
        """Map teleoperation actions to robot joint positions."""
        
        # Create action tensor with zeros for all joints
        action = torch.zeros(len(joint_names), dtype=torch.float32)
        
        try:
            # Extract arm actions (in radians)
            left_arm_actions = actions.get('left_arm', {}).get('qpos', [0.0] * 7)
            right_arm_actions = actions.get('right_arm', {}).get('qpos', [0.0] * 7)
            
            # Extract hand actions (servo values need conversion)
            left_hand_servo = actions.get('left_hand', {}).get('qpos', [700.0] * 6)
            right_hand_servo = actions.get('right_hand', {}).get('qpos', [700.0] * 6)
            
            # Debug first frame only - reduce clutter
            if self.time == 0.0:
                print(f"🔍 First frame debug:")
                print(f"   Left arm: {len(left_arm_actions)} values, Right arm: {len(right_arm_actions)} values")
                print(f"   Left hand: {len(left_hand_servo)} servo values, Right hand: {len(right_hand_servo)} servo values")
            
            # Ensure arm actions have exactly 7 values each
            if len(left_arm_actions) < 7:
                left_arm_actions = list(left_arm_actions) + [0.0] * (7 - len(left_arm_actions))
            elif len(left_arm_actions) > 7:
                left_arm_actions = left_arm_actions[:7]
                
            if len(right_arm_actions) < 7:
                right_arm_actions = list(right_arm_actions) + [0.0] * (7 - len(right_arm_actions))
            elif len(right_arm_actions) > 7:
                right_arm_actions = right_arm_actions[:7]
            
            # Convert hand servo values to radians
            left_hand_radians = self._convert_hand_servo_to_radians(left_hand_servo)
            right_hand_radians = self._convert_hand_servo_to_radians(right_hand_servo)
            
            # Debug first frame only
            if self.time == 0.0:
                print(f"   Converted to radians - Left: {len(left_hand_radians)}, Right: {len(right_hand_radians)} values")
            
            # Map arm joints
            for joint_name, value in zip(['left_arm', 'right_arm'], [left_arm_actions, right_arm_actions]):
                for i, joint_value in enumerate(value):
                    if i < 7:  # Only use first 7 DOF for each arm
                        mapped_key = f"{joint_name}_{i}"
                        # Find this joint in the robot's joint names
                        for robot_joint_idx, robot_joint_name in enumerate(joint_names):
                            if mapped_key in self.arm_joint_mapping.values():
                                # Find the actual joint name
                                for g1_joint, teleop_joint in self.arm_joint_mapping.items():
                                    if teleop_joint == mapped_key:
                                        if g1_joint in robot_joint_name or robot_joint_name.replace('_joint', '') in g1_joint:
                                            action[robot_joint_idx] = joint_value * self.scale_factor
                                            break
            
            # Map hand joints (inspire hand mapping)
            for hand_side, hand_radians in zip(['left', 'right'], [left_hand_radians, right_hand_radians]):
                prefix = 'L_' if hand_side == 'left' else 'R_'
                
                # Ensure we have at least 6 hand values
                if len(hand_radians) < 6:
                    if self.time == 0.0:  # Only warn on first frame
                        print(f"⚠️  Warning: Only {len(hand_radians)} hand radians for {hand_side}, padding")
                    hand_radians = list(hand_radians) + [0.0] * (6 - len(hand_radians))
                
                # Map the 6 hand parameters to 12 inspire joints according to specification
                hand_joint_values = {
                    f'{prefix}index_proximal_joint': hand_radians[0],
                    f'{prefix}index_intermediate_joint': hand_radians[1],
                    f'{prefix}middle_proximal_joint': hand_radians[0],   # Same as index
                    f'{prefix}middle_intermediate_joint': hand_radians[1], # Same as index
                    f'{prefix}pinky_proximal_joint': hand_radians[0],    # Same as index
                    f'{prefix}pinky_intermediate_joint': hand_radians[1], # Same as index
                    f'{prefix}ring_proximal_joint': hand_radians[0],     # Same as index
                    f'{prefix}ring_intermediate_joint': hand_radians[1],  # Same as index
                    f'{prefix}thumb_proximal_yaw_joint': hand_radians[2],
                    f'{prefix}thumb_proximal_pitch_joint': hand_radians[3],
                    f'{prefix}thumb_intermediate_joint': hand_radians[4],
                    f'{prefix}thumb_distal_joint': hand_radians[5]
                }
                
                # Apply to robot joints
                for inspire_joint, value in hand_joint_values.items():
                    for robot_joint_idx, robot_joint_name in enumerate(joint_names):
                        if inspire_joint.lower() in robot_joint_name.lower():
                            action[robot_joint_idx] = value * self.scale_factor
            
        except Exception as e:
            print(f"❌ Error in _map_to_robot_joints: {e}")
            print(f"   Actions structure: {list(actions.keys()) if actions else 'None'}")
            if actions:
                for key, value in actions.items():
                    if isinstance(value, dict) and 'qpos' in value:
                        print(f"   {key}: {len(value['qpos'])} values")
            raise
        
        return action
    
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from current frame of teleoperation data."""
        
        try:
            # Debug first call only - reduce startup clutter
            if self.time == 0.0:
                print(f"🔥 Teleoperation replay starting!")
                print(f"   Device: {env.device}")
                
                if hasattr(env, 'scene'):
                    scene_keys = list(env.scene.keys()) if hasattr(env.scene, 'keys') else []
                    
                    # Access robot directly by key
                    if 'robot' in scene_keys:
                        robot = env.scene['robot']
                        if hasattr(robot, 'data') and hasattr(robot.data, 'joint_names'):
                            joint_names = robot.data.joint_names
                            print(f"   Robot found with {len(joint_names)} joints")
                        else:
                            print(f"   Robot found but no joint_names accessible")
                    else:
                        print(f"❌ No robot found in scene keys: {scene_keys}")
                        return None
                else:
                    print(f"❌ No scene found in environment!")
                    return None
            
            # Check if we should advance to next frame
            if self.time >= self.next_frame_time:
                # Only advance if we haven't reached the end
                if self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                    self.next_frame_time += self.frame_duration / self.playback_speed
                else:
                    # We've reached the end, stop here
                    if int(self.time * 500) % 250 == 0:  # Only show end message occasionally
                        print(f"🏁 Episode complete - holding at final frame {self.current_frame}/{self.total_frames}")
            
            # Check if replay is completed BEFORE trying to get actions
            if self.current_frame >= self.total_frames:
                print(f"🏁 Replay completed! {self.total_frames} frames played")
                return None
            
            # Get current frame actions
            actions = self._get_current_frame_actions()
            if not actions:
                print(f"❌ No actions found for frame {self.current_frame}")
                return None
            
            # Map to robot joint positions
            if hasattr(env, 'scene'):
                scene_keys = list(env.scene.keys()) if hasattr(env.scene, 'keys') else []
                if 'robot' in scene_keys:
                    robot = env.scene['robot']
                    if hasattr(robot, 'data') and hasattr(robot.data, 'joint_names'):
                        joint_names = robot.data.joint_names
                        action = self._map_to_robot_joints(actions, joint_names)
                        
                        # Add batch dimension
                        action = action.unsqueeze(0).to(env.device)
                        
                        # Update time
                        self.time += self.dt
                        
                        # Debug output - show progress every 1 second instead of 0.5 seconds
                        if int(self.time * 500) % 500 == 0:  # Every 1.0 seconds
                            progress_percent = 100 * self.current_frame / self.total_frames
                            remaining_frames = self.total_frames - self.current_frame
                            estimated_time_left = remaining_frames * self.frame_duration / self.playback_speed
                            print(f"🎬 Frame {self.current_frame:4d}/{self.total_frames} | {progress_percent:5.1f}% | ⏱️ {self.time:.1f}s | ETA: {estimated_time_left:.1f}s")
                        
                        return action
                    else:
                        print(f"❌ Robot found but no joint_names accessible!")
                        return None
                else:
                    print(f"❌ No robot found in scene keys: {scene_keys}")
                    return None
            else:
                print(f"❌ No scene found in environment during action generation!")
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
        print("🔄 Teleoperation replay reset")


def create_teleoperation_replay_provider(episode_index: int = 0, **kwargs):
    """Factory function to create teleoperation replay provider."""
    
    episode_path = f"/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data/episode_{episode_index:04d}/data.json"
    
    if not os.path.exists(episode_path):
        available_episodes = []
        data_dir = "/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data"
        if os.path.exists(data_dir):
            available_episodes = [d for d in os.listdir(data_dir) if d.startswith('episode_')]
        raise ValueError(f"Episode {episode_index} not found at {episode_path}. Available episodes: {len(available_episodes)}")
    
    return TeleoperationReplayActionProvider(episode_path, **kwargs)


def main():
    """Test the teleoperation replay provider."""
    print("🎬 Teleoperation Replay Action Provider Test")
    print("=" * 60)
    
    # List available episodes
    data_dir = "/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data"
    if os.path.exists(data_dir):
        episodes = sorted([d for d in os.listdir(data_dir) if d.startswith('episode_')])
        print(f"📁 Found {len(episodes)} episodes: {episodes[:5]}{'...' if len(episodes) > 5 else ''}")
    
    # Test with episode 0
    try:
        provider = create_teleoperation_replay_provider(episode_index=0, scale_factor=1.0, playback_speed=1.0)
        print(f"✅ Successfully created replay provider")
        print(f"   📊 {provider.total_frames} frames loaded")
        print(f"   ⏱️  Duration: {provider.total_frames * provider.frame_duration:.1f}s")
        
    except Exception as e:
        print(f"❌ Error creating provider: {e}")


if __name__ == "__main__":
    main()
