# Super Mario Bros Gym Environment Setup

## Requirements
- **Python**: 3.8
- **CUDA**: 12.1 (for GPU support)

## Installation
Install the required packages using the following commands:

```bash
pip install nes_py
pip install gym_super_mario_bros
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install keyboard
```

## GPU Test
After installation, verify GPU support by running:

```bash
python src/test/gpu_test.py
```

## Fix for Gym Environment Error
Some users may encounter the following error when running the Mario environment:

```
File "**\site-packages\gym\wrappers\time_limit.py", line 50, in step
observation, reward, terminated, truncated, info = self.env.step(action)
ValueError: not enough values to unpack (expected 5, got 4)
```

### **Solution**
Edit the error line inside `time_limit.py` to the following:

```python
observation, reward, terminated, info = self.env.step(action)
self._elapsed_steps += 1
truncated = False
if self._elapsed_steps >= self._max_episode_steps:
    truncated = True

return observation, reward, terminated, truncated, info
```

This ensures compatibility with the current gym environment API.

## Notes
- Ensure you are using **Python 3.8**, as other versions may cause compatibility issues.
- If additional issues arise, consider updating `gym` or checking community forums for further fixes.

