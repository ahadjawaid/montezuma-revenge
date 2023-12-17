from visualize import visualize_episode
from utils import get_device
from utils import get_sorted_checkpoints

env_name = "ALE/Assault-v5"

checkpoint_path = get_sorted_checkpoints("checkpoints")[-1]
visualize_episode(env_name, checkpoint_path=checkpoint_path, device=get_device())