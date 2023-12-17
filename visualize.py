from utils import get_device
from PIL import Image
from trainer import DQNTrainer
import torch
import gym
from gym.wrappers import TimeLimit


def visualize_episode(
        checkpoint_path: str = None, 
        trainer=None, 
        device: str = None
    ):
    assert checkpoint_path is not None or trainer is not None, "Either checkpoint_path or trainer must be provided"
    device = get_device(device)

    if trainer is None and checkpoint_path:
        trainer = DQNTrainer.load(checkpoint_path)
    env_name = trainer.env_name
    env = gym.make(env_name, render_mode="human")

    state = torch.from_numpy(env.reset()[0]).float().unsqueeze(0).to(device)
    done = truncated = False
    while not done and not truncated:
        env.render()
        q_values = trainer.online_model(state)
        action = q_values.argmax(dim=1)
        # action = trainer.exploration.get_action(q_values)
        state, _, done, truncated, _ = env.step(action)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    print(f"Done: {done}, Truncated: {truncated}")
    env.close()


def get_image(observation):
    img = Image.fromarray(observation)
    return img
