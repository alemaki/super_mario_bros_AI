import numpy as np
import torch

"""Moves data to device, downsamples, crops, grayscales and normalises"""
def preprocess_state(state: np.ndarray, device) -> torch.Tensor:
    state = torch.tensor(np.ascontiguousarray(state), device=device, dtype=torch.float32)
    state = torch.mean(state, dim=2).byte()
    state = state[32:240:2, ::2]
    return state / 255.0