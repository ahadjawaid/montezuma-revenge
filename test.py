from visualize import visualize_episode
from utils import get_device
from utils import get_sorted_checkpoints

checkpoint_path = get_sorted_checkpoints("checkpoints")[-1]
visualize_episode(checkpoint_path=checkpoint_path, device=get_device())