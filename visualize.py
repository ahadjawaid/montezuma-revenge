from utils import get_device
from PIL import Image
from trainer import DQNTrainer
import torch
import gym


def visualize_episode(
        env_name: str, 
        checkpoint_path: str = None, 
        model=None, 
        model_params: dict = None, 
        device: str = None
    ):
    device = get_device(device)

    env = gym.make(env_name, render_mode="human")
    if model is None:
        trainer = DQNTrainer.load(checkpoint_path)

    state = torch.from_numpy(env.reset()[0]).float().unsqueeze(0).to(device)
    done = False
    while not done:
        env.render()
        q_values = trainer.online_model(state)
        action = q_values.argmax(dim=1)
        state, _, done, _, _ = env.step(action)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    env.close()


def get_image(observation):
    img = Image.fromarray(observation)
    return img
