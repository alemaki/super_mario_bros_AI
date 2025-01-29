from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# endviromental variables 

#needs python 3.8
done = True
printed = True 
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, _, info = env.step(env.action_space.sample())
    env.render()

    if not printed:
        print(f"{state}, {type(state)}, {state.shape}, {state.size}, {reward}, {done}, {info}"
        )
        printed = True
        break

env.close()