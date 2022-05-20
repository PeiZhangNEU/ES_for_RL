import gym
import config
import model as md
from es import *

game = config.games['gym_cartpole']
model = md.make_model(game)
num_params = model.param_count
population = 256
print("size of model", num_params)
train_mode_int = 1
seed = 123
max_len = 1000
num_episode = 2

oes = OpenES(num_params,                  # number of model parameters
            sigma_init=0.5,            # initial standard deviation
            sigma_limit=0.1,
            sigma_decay=0.999,         # don't anneal standard deviation
            learning_rate=0.1,         # learning rate for standard deviation
            learning_rate_decay = 0.9, # annealing the learning rate
            learning_rate_limit= 0.001,
            popsize=population,       # population size
            antithetic=False,          # whether to use antithetic sampling
            weight_decay=0.00,         # weight decay coefficient
            rank_fitness=False,        # use rank rather than fitness numbers
            forget_best=False)
es = oes

# 所有关于env的东西都放在了model类中
model.make_env(seed=seed, render_mode=False)  # 这里的render mode 是打印赋予的参数

print(model.env.action_space)
print(model.env.observation_space)

history_rew = []
for i in range(10000):
    print('iter:',i )
    print('learning_rate', oes.learning_rate)
    solutions = es.ask()
    fitness_list = []
    for solution in solutions:
        train_mode = (train_mode_int == 1)
        model.set_model_params(solution)
        reward_list, t_list = md.simulate(model,
                    train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
        # reward = np.min(reward_list)  # 奖励选择方式很多，可以用最小也可以用平均
        reward = np.mean(reward_list)
        fitness_list.append(reward)
    es.tell(fitness_list)
    result = es.result()
    history_rew.append(result[1])
    if (i+1) % 2==0:
        print('reward at itr', (i+1), result[1])
        np.save('cartpole_nn_params.npy', result[0])
print('local best nn params:\n', result[0])
print('reward best is:', result[1])
