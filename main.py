import os
import time
import pickle
from pathlib import Path

from datasets import *
from simulator import Simulator
from agents.toy_models import *
from agents.drqn_kbc import DRQN_KBC


N_GAMES = int(3e1)      # number of total games
T = 20                  # timesteps for a game
SWITCH_THRES = 20       # switch threshold for IS and KA 

# set up data path 
DATA_PATH = Path.cwd() / 'data'
print('DATA_PATH: {}'.format(DATA_PATH))

# load and process the dataset
names = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'UMLS', 'KINSHIP', 'NATIONS']
name = names[4]
dataset = dataset_to_dict(os.path.join(DATA_PATH, name), 'train.pickle')
dataset_agent = dataset_split(dataset)
dataset = ExternalKB(dataset)
dataset_agent = InternalKB(dataset_agent)
print("Dataset: {} with {} Entities and {} Predicates. ".format(name, dataset.n_entities, dataset.n_predicates))

# set up log path 
exp_name = '{}_{}'.format(name, time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = Path.cwd() / 'log' / exp_name
assert not os.path.exists(exp_dir), \
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
        exp_dir)
os.makedirs(exp_dir, exist_ok=True)

# instantiate an agent 
agent = AgentRandom(dataset_agent, T, SWITCH_THRES)
# instantiate the simulator
simulator = Simulator(dataset)

print(" \n### Game start ####")
win_log = []
for i in range(N_GAMES):
	simulator.entity_select()
	agent.reset()
	while 1:
		q, _, done = agent.question()
		r = simulator.response(q)
		agent.update_response(r)
		if done:
			break

	guess = agent.guess_generate()
	right, answer = simulator.guess_check(guess, verbose=False)
	agent.get_feedback(right)
	win_log.append(int(right))
	print("Game {} - Guess/Target: {:>11} Victory: {}".format(i, str(guess) + '/' + str(answer), right))


