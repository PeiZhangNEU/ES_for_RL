import gym
import config
import model as md
from es import *

game = config.games['mujoco_hopper']
model = md.make_model(game)
num_params = model.param_count
population = 256
print("size of model", num_params)
train_mode_int = 0
seed = 123
max_len = 1000
num_episode = 5

# 所有关于env的东西都放在了model类中
model.make_env(seed=seed, render_mode=False)  # 这里的render mode 是打印赋予的参数

print(model.env.action_space)
print(model.env.observation_space)


best_params = np.load('hopper_nn_params.npy')
train_mode = (train_mode_int == 1)
model.set_model_params(best_params)
reward_list, t_list = md.simulate(model,
            train_mode=train_mode, render_mode=True, num_episode=num_episode, seed=seed, max_len=max_len)
reward = np.mean(reward_list)  # 奖励选择方式很多，可以用最小也可以用平均
print(reward)
    
