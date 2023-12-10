# Montezuma Revenge

We use the open ai gym environment for Montezuma Revenge. The environment is a 2D platformer where the agent has to navigate through a maze and collect keys to unlock doors. The agent has to avoid enemies and traps. The agent can jump, climb ladders, and use ropes to navigate the environment. The agent receives a reward of 1 for collecting a key and 10 for unlocking a door. The agent receives a reward of -1 for dying. The agent receives a reward of 0 for all other actions. The agent receives a reward of 100 for completing the game.

## Getting Started

```
git clone https://github.com/ahadjawaid/montezuma-revenge.git
```
```
cd montezuma-revenge
```

### Create Environment (Optional)

```
conda create -n montezuma-revenge
```

or 

```
python3 -m venv montezuma-revenge
```

### Installing Dependencies

```
pip install -r requirements.txt
```

