import numpy as np
import gymnasium as gym
import gym_pusht
import zarr

from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset_path = "/network/scratch/o/ozgur.aslan/data/pusht/pusht_cchi_v7_replay.zarr"
dataset_root = zarr.open(dataset_path, 'r')
# All demonstration episodes are concatinated in the first dimension N
train_data = {
    # (N, action_dim)
    'action': dataset_root['data']['action'][:],
    # (N, obs_dim)
    'obs': dataset_root['data']['state'][:]
}
# Marks one-past the last index for each episode
episode_ends = dataset_root['meta']['episode_ends'][:]

new_features = {"observation.image": {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "observation.state": {'dtype': 'float64', 'shape': (7, ), 'names': [["agent_pos_x", "agent_pos_y", "agent_vel_x", "agent_vel_y", "block_pos_x", "block_pos_y", "block_angle"]]},
                "action": {'dtype': 'float32', 'shape': (2, ), 'names': None}
            }
new_dataset = LeRobotDataset.create("ozgraslan/pusht_224", fps=10, use_videos=True , features=new_features)
print(new_dataset.meta.features)

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=224, observation_height=224, render_mode="rgb_array")
episode_starts = [0] + list(episode_ends[:-1])
for ep_id in range(len(episode_ends)):
    start_idx = episode_starts[ep_id]
    end_idx = episode_ends[ep_id]
    start_state = train_data["obs"][start_idx]
    actions = train_data["action"][start_idx:end_idx]
    obs, info = env.reset(options={"reset_to_state": start_state})
    print(f"Resetting env for episode {ep_id} with start state {start_state}")
    for action in actions:
        frame = {
            "observation.image": obs['pixels'],
            "observation.state": np.concatenate([info['pos_agent'], info["vel_agent"], info["block_pose"]]) ,
            "action": action,
            "task": "pusht"
        }
        new_dataset.add_frame(frame)
        obs, reward, terminated, truncated, info = env.step(action)
    new_dataset.save_episode()

new_dataset.finalize()
new_dataset.push_to_hub(tags=["pusht", "224x224"], private=False)