import gym
import torch.nn as nn
import torch
import numpy as np
import random
from typing import Callable
from exploration import EpsilonGreedyExploration
from buffer import ReplayBuffer
from torch.optim import Optimizer
from tqdm.auto import tqdm
from utils import get_device
from pathlib import Path
from gym.wrappers import TimeLimit


class DQNTrainer:
    def __init__(
            self, 
            env_name: str, 
            Model: nn.Module,
            model_params: dict,
            exploration: EpsilonGreedyExploration, 
            Buffer: ReplayBuffer, 
            buffer_params: dict,
            discount_rate: float,
            loss_fn: Callable,
            Optim: Optimizer,
            max_steps: int = 10000,
            lr: float = 1e-4,
            network_frozen_steps: int = 1000,
            seed: int = 23,
            time_step_reward: float = 0,
            save_interval: int = 10000,
            max_episode_steps: int = 500,
            debug: bool = False,
            device: str = None,
            **kwargs
        ):
        self.env_name = env_name
        self.env = TimeLimit(gym.make(env_name), max_episode_steps)
        self.state, _ = self.env.reset()
        self.device = get_device(device)
        self.model_params = model_params
        model_params["in_size"] = self.state.shape[:2]
        self.model_args = dict(out_dim=self.env.action_space.n, **model_params)
        self.Model = Model
        self.lr = lr
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.online_model = Model(**self.model_args).to(self.device)
        self.target_model = Model(**self.model_args).to(self.device)
        self.discount_rate = discount_rate
        self.network_frozen_steps = network_frozen_steps
        self.loss_fn = loss_fn
        self.Optim = Optim
        self.optimizer = Optim(self.online_model.parameters(), lr=lr, weight_decay=kwargs.get("weight_decay", 0))
        self.exploration = exploration
        self.Buffer = Buffer
        self.buffer_params = buffer_params
        self.replay_buffer = Buffer(**buffer_params)
        self.time_step_reward = time_step_reward
        self.debug = debug
        self.loss_history = []
        self.seed = seed
        self.step = 0
        self._set_seed(seed)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)

    def _print_debug_statement(self, item):
        if self.debug:
            print(item)

    def _calculate_value_loss(self, batch: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), batch)
        rewards, dones = rewards.squeeze(), dones.squeeze()

        values = self.online_model(states).gather(1, actions.long()).squeeze()
        next_values = self.target_model(next_states).argmax(dim=1).detach()

        expected_values = rewards + self.discount_rate * next_values * (1 - dones)

        loss = self.loss_fn(values, expected_values)

        return loss
    
    def _generate_experiences(self, state: torch.Tensor, step: int):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            
        action_values = self.online_model(state.to(self.device))
        action = self.exploration.get_action(action_values)

        next_state, reward, done, truncated, info = self.env.step(action)
        self._print_debug_statement(f"Step: {step}, Info: {info}")

        reward = self.time_step_reward if reward == 0 else reward

        self.replay_buffer.add((
            state.squeeze(), 
            torch.tensor([action]).float(), 
            torch.tensor([reward]).float(), 
            torch.from_numpy(next_state).float(), 
            torch.tensor([done]).float()
        ))

        return state, action, reward, next_state, truncated, done

    def _fit_one_step(self):
        if self.replay_buffer.is_ready():
            batch = self.replay_buffer.sample()
            value_loss = self._calculate_value_loss(batch)
            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()

            return value_loss.item()
        
    def fit(self):
        state = torch.from_numpy(self.state).float()

        progress_bar = tqdm(total=self.max_steps, desc="Training", unit="step")
        range_steps = range(self.step, self.max_steps)
        for step in range_steps:
            self.step = step
            state, _, _, _, truncated, done = self._generate_experiences(state, step)

            if step % self.network_frozen_steps == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())

            if done or truncated:
                state, _ = self.env.reset()
                state = torch.from_numpy(state).float()

            value_loss = self._fit_one_step()
            self.loss_history.append(value_loss)
            
            progress_bar.update(1)
            progress_bar.set_postfix({"value_loss": value_loss}, refresh=True)

            if ((step % self.save_interval == 0 or step % (self.max_steps // 10) == 0) and step > 0) or step == self.max_steps - 1:
                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)

                checkpoints = list(checkpoint_dir.glob("*.pt"))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: x.stat().st_mtime)

                    for checkpoint in checkpoints[:-2]:
                        checkpoint.unlink()

                self.save(checkpoint_dir / f"{step}.pt")
        
    def save(self, path: str) -> None:
        checkpoint = {
            "online_model": self.online_model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "Model": self.Model,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "Optim": self.Optim,
            "network_frozen_steps": self.network_frozen_steps,
            "optimizer": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "exploration": self.exploration,
            "Buffer": self.Buffer,
            "buffer_params": self.buffer_params,
            "discount_rate": self.discount_rate,
            "time_step_reward": self.time_step_reward,
            "loss_fn": self.loss_fn,
            "model_args": self.model_args,
            "model_params": self.model_params,
            "env_name": self.env_name,
            "seed": self.seed,
            "device": self.device,
            "step": self.step
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> None:
        checkpoint = torch.load(path)
        online_state_dict = checkpoint.pop("online_model")
        target_state_dict = checkpoint.pop("target_model")
        optimizer_state_dict = checkpoint.pop("optimizer")
        loss_history = checkpoint.pop("loss_history")

        trainer = cls(**checkpoint)
        trainer.online_model.load_state_dict(online_state_dict)
        trainer.target_model.load_state_dict(target_state_dict)
        trainer.optimizer.load_state_dict(optimizer_state_dict)
        trainer.loss_history = loss_history

        trainer.step = checkpoint["step"]

        return trainer
    
class DDQNTrainer(DQNTrainer):
    def _calculate_value_loss(self, batch: torch.Tensor) -> torch.Tensor:
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), batch)
        rewards, dones = rewards.squeeze(), dones.squeeze()

        q_values = self.online_model(states)
        values = q_values.gather(1, actions.long()).squeeze()

        online_actions = q_values.argmax(dim=1).unsqueeze(1)
        target_q_values = self.target_model(next_states)
        next_values = target_q_values.gather(1, online_actions.long()).squeeze().detach()

        expected_values = rewards + self.discount_rate * next_values * (1 - dones)

        loss = self.loss_fn(values, expected_values)

        return loss