import os
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import gc

from datasets import *
from simulator import Simulator
from agents.toy_models import *
from agents.drqn_kbc import AgentDQN, AgentDRQN


N_GAMES = int(1e4)      # number of total games
T = 20                  # timesteps for a game
SWITCH_THRES = 20       # switch threshold for IS and KA
N_AVERAGE = 100        # number of average games

# set up data path
DATA_PATH = Path.cwd() / 'data'
print('DATA_PATH: {}'.format(DATA_PATH))

# load and process the dataset
dataset_collection = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'UMLS', 'KINSHIP', 'NATIONS']
name = dataset_collection[7]
dataset = dataset_to_dict(os.path.join(DATA_PATH, name), 'train.pickle')
dataset_agent = dataset_split(dataset)
dataset = ExternalKB(dataset)
dataset_agent = InternalKB(dataset_agent)
print("Dataset: {} with {} Entities and {} Predicates. ".format(name, dataset.n_entities, dataset.n_predicates))

# select agent
agent_collection = ["Random", "DQN", "DRQN"]
agent_name = agent_collection[2]

# set up log path 
exp_name = '{}_{}_{}'.format(name, agent_name, time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = Path.cwd() / 'log' / exp_name
assert not os.path.exists(exp_dir), \
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
        exp_dir)
os.makedirs(exp_dir, exist_ok=True)

# set up pre-train load path
load_path = Path.cwd() / 'log' /'NATIONS_DRQN_13-07-2020_21-51-46'

# instantiate agent
if agent_name == "Random":
	agent = AgentRandom(dataset_agent, SWITCH_THRES, T, load_path=load_path)
elif agent_name == "DQN":
	agent = AgentDQN(dataset_agent, SWITCH_THRES, T, load_path=load_path)
elif agent_name == "DRQN":
	agent = AgentDRQN(dataset_agent, SWITCH_THRES, T, load_path=load_path)
else:
	raise RuntimeError('No match agent is found!')

# instantiate simulator
simulator = Simulator(dataset)

print(" \n### Game start ####")
win_log = []
win_curve = []
for i in range(N_GAMES):
	simulator.entity_select()
	agent.reset()
	while 1:
		question, switch, done = agent.question()
		if done:
			break
		response = simulator.response(question)
		agent.update_response(response)
		# print(q, switch)

		if switch:
			guess, prob = agent.guess_generate(normalize=True)
			right, answer = simulator.guess_check(guess, verbose=False)
			agent.get_feedback(right)
	win_log.append(int(right))
	# print("Game {} - Guess/Target: {:>11} Confidence: {:4f} Victory: {}".format(i, str(guess) + '/' + str(answer), prob, right))
	# exit()
	# save logs
	if (i+1) % N_AVERAGE == 0:
		# record winning rate
		success_rate = np.average(np.array(win_log))
		print("Game {}/{} - Success rate: {}".format(i + 1, N_GAMES, success_rate))
		win_curve.append(success_rate)
		fig = plt.figure()
		plt.plot(N_GAMES*np.arange(len(win_curve)), win_curve)
		plt.xlabel("Game Steps")
		plt.ylabel("Winning Rate")
		plt.savefig(os.path.join(exp_dir, 'win_curve.png'))
		win_log_l = [str(i) for i in win_log]
		string = ' '.join(win_log_l)
		with open(os.path.join(exp_dir, 'raw.txt'), "a") as myfile:
			myfile.write(string)
		win_log = []
		# save model
		agent.save_model(exp_dir)
		gc.collect()


