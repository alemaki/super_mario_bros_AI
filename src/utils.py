import numpy as np
import torch
import csv
import os

"""Moves data to device, downsamples, crops, grayscales and normalises"""
def preprocess_state(state: np.ndarray, device) -> torch.Tensor:
    state = torch.tensor(np.ascontiguousarray(state), device=device, dtype=torch.float32)
    state = state.mean(dim=2)
    state = state[32:240:2, ::2]
    return state / 255.0


def preprocess_smaller_state(state: np.ndarray, device) -> torch.Tensor:
    state = torch.tensor(np.ascontiguousarray(state), device=device, dtype=torch.float32)
    state = state.mean(dim=2)
    state = state[50:220:2, 30:190:2] #  less info
    return state / 255.0

def record_info_for_episode(file_name, episode, total_reward, time, level_data):
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Time', 'Reward', 'World', 'Stage', 'X Position', 'Y Position', 'Life', 'Score', 'Status', 'Reached Flag'])

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            episode,
            time,
            total_reward, 
            level_data['world'],
            level_data['stage'],
            level_data['x_pos'], 
            level_data['y_pos'],
            level_data['life'],
            level_data['score'],
            level_data['status'],
            level_data['flag_get'],
        ])


def record_info_for_worker(file_name, episode, worker_id, total_reward, time, level_data):
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Time', 'Reward', 'World', 'Stage', 'X Position', 'Y Position', 'Life', 'Score', 'Status', 'Reached Flag'])

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            episode,
            worker_id,
            time,
            total_reward, 
            level_data['world'],
            level_data['stage'],
            level_data['x_pos'], 
            level_data['y_pos'],
            level_data['life'],
            level_data['score'],
            level_data['status'],
            level_data['flag_get'],
        ])