import os
import pickle
from pathlib import Path

from simulator import Simulator
from agents.toy_models import *
from agents.drqn_kbc import DRQN_KBC
from utils import dataset_to_tensor

DATA_PATH = Path.cwd() / 'data'
print('DATA_PATH: {}'.format(DATA_PATH))

N_GAMES = int(1e5)   # number of total games
T = 20               # timesteps for a game

# load and process the dataset
name = 'FB237'
name = 'NATIONS'
dataset = dataset_to_tensor(os.path.join(DATA_PATH, name), 'train.pickle')
dataset_t = dataset['tensor']
print(dataset_t.shape)

# define agent and simulator
agent = DRQN_KBC()
simulator = Simulator(dataset)

for i in range(N_GAMES):
	target = simulator.entity_select()
	agent.reset()
	for t in range(T):
		q = agent.question()
		a = simulator.response()
		agent.update_response(a)

	guess = agent.guess_generate()
	right, answer = guess_check(guess, verbose=False)
