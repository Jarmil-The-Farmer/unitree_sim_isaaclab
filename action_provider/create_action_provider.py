from pathlib import Path
import os

from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_teleoperation import TeleoperationReplayActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.action_provider_diffusion import DiffusionActionProvider


def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        return DDSActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env,args_cli=args)
    elif args.action_source == "teleoperation_replay":
        data_dir = os.environ["DATASET_ROOT_PATH"]
        return TeleoperationReplayActionProvider(data_dir, episode_index=args.episode_index, playback_speed=args.playback_speed)
    elif args.action_source == "diffusion":
        data_dir = os.environ["DATASET_ROOT_PATH"]
        return DiffusionActionProvider(data_dir, episode_index=args.episode_index, checkpoint_path=args.checkpoint_path)
    else:
        print(f"unknown action source: {args.action_source}")
        return None