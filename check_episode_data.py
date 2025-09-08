#!/usr/bin/env python3
"""
Check episode data structure to debug frame count issue
"""

import json
import os

def check_episode_data():
    """Check the structure and frame count of episode data."""
    
    # Check multiple episodes
    episodes_to_check = [0, 1]
    
    for episode_idx in episodes_to_check:
        episode_path = f'/home/vaclav/IsaacLab_unitree/IsaacLab/Pretrained_data/data/episode_{episode_idx:04d}/data.json'
        
        print(f"\n{'='*60}")
        print(f"📁 CHECKING EPISODE {episode_idx}")
        print(f"📂 Path: {episode_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(episode_path):
            print(f"❌ Episode {episode_idx} file not found!")
            continue
            
        try:
            with open(episode_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Successfully loaded JSON file")
            print(f"🔍 Top-level data keys: {list(data.keys())}")
            
            # Check if 'data' key exists
            if 'data' in data:
                episode_frames = data['data']
                print(f"📊 Episode {episode_idx} has {len(episode_frames)} frames in 'data' key")
            else:
                print(f"⚠️  No 'data' key found! Using entire JSON as frames")
                episode_frames = data if isinstance(data, list) else []
                print(f"📊 Episode {episode_idx} has {len(episode_frames)} frames (direct)")
            
            if episode_frames and len(episode_frames) > 0:
                # Check first frame structure
                first_frame = episode_frames[0]
                print(f"🔍 First frame keys: {list(first_frame.keys())}")
                
                if 'actions' in first_frame:
                    actions = first_frame['actions']
                    print(f"🔍 First frame actions keys: {list(actions.keys())}")
                    
                    # Check each action type
                    for action_key in ['left_arm', 'right_arm', 'left_hand', 'right_hand']:
                        if action_key in actions:
                            action_data = actions[action_key]
                            if isinstance(action_data, dict) and 'qpos' in action_data:
                                qpos = action_data['qpos']
                                print(f"   {action_key}: {len(qpos)} values - {qpos[:3]}..." if len(qpos) > 3 else f"   {action_key}: {qpos}")
                            else:
                                print(f"   {action_key}: {action_data}")
                else:
                    print(f"⚠️  No 'actions' key in first frame!")
                
                # Check last frame
                if len(episode_frames) > 1:
                    last_frame = episode_frames[-1]
                    print(f"🔍 Last frame (#{len(episode_frames)-1}) keys: {list(last_frame.keys())}")
                
                # Check file size
                file_size = os.path.getsize(episode_path)
                print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                
            else:
                print(f"❌ No frames found in episode {episode_idx}!")
                
        except Exception as e:
            print(f"❌ Error loading episode {episode_idx}: {e}")

if __name__ == "__main__":
    check_episode_data()
