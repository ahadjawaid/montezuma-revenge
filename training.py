import torch.nn as nn
from model import DQN
from trainer import DQNTrainer, DDQNTrainer
from exploration import EpsilonGreedyExploration, quadratic_decay_schedule
from buffer import ReplayBuffer
import torch
from pathlib import Path

def main(env_name, max_steps, save_dir, Trainer):
    initial_epsilon = 1.0
    batch_size = 32

    params = dict(
        env_name=env_name, 
        Model=DQN,
        model_params=dict(
            conv_feats=128,
            hidden_dim = 48,
            n_layers = 2,
            Activation = nn.ReLU,
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
        time_step_reward=0.0,
        network_frozen_steps=100,
        seed = 42,
        save_interval = 1000,
        weight_decay = 1e-3,
        max_steps = max_steps,
        max_episode_steps=10000,
        debug = False,
    )

    trainer = Trainer(**params)
    trainer.fit()

    model_dir = Path(save_dir)
    model_dir.mkdir(exist_ok=True)
    torch.save(trainer.online_model.state_dict(), model_dir/"model.pt")

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/KungFuMaster-v5")
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--trainer", type=str, default="DQN")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    env_name = args.env
    max_steps = args.max_steps
    save_dir = args.save_dir
    Trainer = DQNTrainer if args.model == "DQN" else DDQNTrainer if args.model == "DDQN" else None
    assert Trainer is not None, "Invalid trainer"

    main(env_name, max_steps, save_dir, Trainer)
