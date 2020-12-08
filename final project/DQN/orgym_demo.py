import or_gym
import numpy as np
env = or_gym.make('BinPacking-v0')
done=True
while  done:
    state, reward, done, _ = env.step(1)
