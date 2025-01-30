import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
from utils import preprocess_state

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Preprocess the state
state = preprocess_state(env.reset(), device='cuda')

# Convert tensor to numpy for visualization
state_np = state.cpu().numpy()

# Display the resulting image
plt.imshow(state_np, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()