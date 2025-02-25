from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "../models"

@dataclass
class BaseConfig:
    LEVEL: str = 'SuperMarioBros-v0'
    SAVE_DIR: Path = MODELS_DIR
    LOG_FILE_NAME: Path = SAVE_DIR / "episodes_log.log"
    LEARNING_RATE: float = 1e-3
    GAMMA: float = 0.99
    ONE_LIFE: bool = True
    MODEL_SAVE_EPISODES: int = 1000  # Common save interval


    def __post_init__(self):
        self.LOG_FILE_NAME = self.SAVE_DIR / "episodes_log.log"
        os.makedirs(self.SAVE_DIR, exist_ok=True)  # Ensure directory exists

@dataclass
class DQNConfig(BaseConfig):
    SAVE_DIR = MODELS_DIR / "dqn_simple_movement_one_life_action_steps_models"
    START_MODEL_EPISODE: int = 12000
    EPISODE_STOP: int = 50000
    EPISODE_SAVE: int = 2000
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.001
    EPSILON_DECAY: float = 0.9995
    EPSILON_UPDATE: int = 20
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 80000
    TARGET_UPDATE: int = 6000
    MAX_STEPS: int = 8000
    ACTION_STEPS: int = 4
    CHANNEL_MULTIPLIER: int = 1

@dataclass
class A2CConfig(BaseConfig):
    SAVE_DIR = MODELS_DIR / "a2c_simple_movement_new_models"
    LOAD_MODEL_EPISODE: int = -1
    MAX_ENV_STEPS: int = 6000
    MAX_EPISODE_TRAIN: int = 10000

@dataclass
class A3CConfig(BaseConfig):
    SAVE_DIR = MODELS_DIR / "a3c_simple_movement_models"
    LOAD_MODEL_EPISODE: int = -1
    LEARNING_RATE: float = 1e-4  # Different from BaseConfig
    ENTROPY_COEF: float = 0.05
    N_STEPS: int = 12
    ACTION_STEPS: int = 4
    MAX_STEPS_ENV: int = 5000
    MAX_EPISODES: int = 50000
