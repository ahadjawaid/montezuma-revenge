import torch.nn as nn
from model import DQN
from trainer import DQNTrainer
from exploration import EpsilonGreedyExploration, quadratic_decay_schedule
from buffer import ReplayBuffer
import torch
from pathlib import Path

env_name = "ALE/MontezumaRevenge-v5"
max_steps = 1000000
initial_epsilon = 1.0
batch_size = 4

params = dict(
    env_name=env_name, 
    Model=DQN,
    model_params=dict(
        conv_feats=256,
        hidden_dim = 128,
        n_layers = 12,
        Activation = nn.ReLU,
        Norm = nn.LayerNorm,
    ),
    exploration=EpsilonGreedyExploration(
        epsilon=initial_epsilon,
        decay_schedule=quadratic_decay_schedule(
            initial_epsilon=initial_epsilon,
            final_epsilon=0.3,
            max_step=max_steps,
        )
    ), 
    Buffer=ReplayBuffer, 
    buffer_params=dict(
        batch_size=batch_size,
        min_size=320,
        max_size=max_steps // 50,
    ),
    discount_rate=0.9,
    loss_fn=torch.nn.SmoothL1Loss(beta=1.0),
    Optim=torch.optim.RMSprop,
    lr=1e-5,
    time_step_reward=-1.0,
    network_frozen_steps=1000,
    seed = 42,
    max_steps = max_steps,
    debug = False
)

trainer = DQNTrainer(**params)

trainer.fit()

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
torch.save(trainer.model.state_dict(), model_dir/"model.pt")