import os, sys
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import logging
import math
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from datasets import *
from utils import *
from simulator import Simulator
from agents.toy_models import *
from agents.wolpdpg_ps import init_parser, AgentWolDDPGSA
from external_agents.Bases.utils.hyperparameters import ConfigEmpty
from agents.embedder import KBC_Embed, Embed_Config

import wandb

# assign configurations
config = ConfigEmpty()

# assign gpu if it available
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	print("CUDA enabled")
else:
    print("CUDA not enabled, force exit")
    # exit()

# random seed
config.seed = 30  #23

# append tunable hyperparams here
# NATIONS
hyperparameter_defaults = dict(
nb_action = 32,   
k_ratio = 0.05,    
hidden1_a = 32,  
hidden2_a = 32,  
hidden1_c = 32, 
hidden2_c = 16, 
bsize = 32,      
rmsize = int(1e4),      
c_lr = 0.0001,       
p_lr = 0.0001,          
epsilon = int(5e5),     
normalize = False,
ou_sigma = 1.0,
intr_reward = False,
is_update = False,       
)

# # UMLS
# hyperparameter_defaults = dict(
# nb_action = 64,  
# k_ratio = 0.01,       
# hidden1_a = 128,  
# hidden2_a = 128, 
# hidden1_c = 128, 
# hidden2_c = 64, 
# bsize = 64,           
# rmsize = int(2e5),   
# c_lr = 0.0001, 
# p_lr = 0.0001,
# epsilon = int(1e6),
# normalize = False,
# ou_sigma = 2.0,        
# intr_reward = True,
# )

########  initiate wandb ########
wandb.init(config=hyperparameter_defaults, project="QsOverKG_2")
d = wandb.config.__dict__
config.add_attr(hyperparameter_defaults)

N_GAMES = int(2e5)      # number of total episodes
T = 20                  # timesteps for a game
SWITCH_THRES = [12, 18] # switch threshold for IS and KA (int for fixed, list for low and upper bound)
N_AVERAGE = 1000        # number of average games
KB_INIT_RATIO = 0.8     # initialization ratio for external KB to internal KB

########  set up data path ########
DATA_PATH = os.path.join(parentdir, 'data')
print('DATA_PATH: {}'.format(DATA_PATH))

########  load and process the dataset ######## 
dataset_collection = ['NATIONS', 'UMLS']
name = dataset_collection[0]
dataset = dataset_to_dict(os.path.join(DATA_PATH, name), ['train.pickle', 'valid.pickle', 'test.pickle'])
dataset_agent, dataset_agent_np = dataset_split(dataset, KB_INIT_RATIO )
dataset = ExternalKB(dataset)
dataset_agent = InternalKB(dataset_agent)
print("Dataset: {} with {} Entities, {} Predicates and {} Entries. ".format(name, dataset.n_entities, dataset.n_predicates, dataset.n_examples))

########  set up log path ########
exp_dir = os.path.join(wandb.run.dir, 'log')
assert not os.path.exists(exp_dir), \
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
        exp_dir)
os.makedirs(exp_dir, exist_ok=True)

######## set up pre-trained load path ######## 
load_path = None
# run_name = 'run-20200924_012009-2uix06el'
# load_path = os.path.join('wandb', run_name, 'log')

######## instantiate agent ########
agent_collection = ["Random", "DQN", "DRQN", "WolDDPG", "LAB-DQN", "LAB-DRQN", "LAB-WolDDPG"]
agent_name = agent_collection[6]
print("Employ Agent {}".format(agent_name))

wol_args = init_parser().parse_args()
config.train_args = ConfigEmpty()
config.train_args.add_attr(vars(wol_args))
d = wandb.config.__dict__
config.train_args.add_attr(d['_items'])

agent = AgentWolDDPGSA(dataset_agent, SWITCH_THRES, config, T, load_path=load_path, train_mode=True, is_update=config.is_update, exp_path=exp_dir)

######## instantiate simulator ########
simulator = Simulator(dataset)

######## set random seed manually ########
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
np.random.seed(123)

######## Main Game ######## 
print(" \n### Game start ####")
win_log = []
win_curve = []
kbr_curve = []
best_success_rate = 0.
kbr_curve.append(KB_completion_ratio(simulator.get_fact_set(), agent.kb.get_fact_set()))

for i in range(N_GAMES):
    simulator.entity_select()
    agent.reset()
    while 1:
        question, switch, done = agent.question()
        if done:
            break
        response = simulator.response(question)
        agent.update_response(response)

        if switch:
            guess, prob = agent.guess_generate(normalize=True)
            right, answer = simulator.guess_check(guess, verbose=False)
            agent.get_feedback(right, prob)
    win_log.append(int(right))

    # record  metrics
    if (i+1) % N_AVERAGE == 0:
        kbc_ratio = KB_completion_ratio(simulator.get_fact_set(), agent.kb.get_fact_set())
        kbr_curve.append(kbc_ratio)
        success_rate, win_curve = metric_logging(win_log, win_curve, exp_dir, i, N_GAMES, N_AVERAGE, T, kbr_curve)
        wandb.log({"winning rate": success_rate, "KBR": kbc_ratio})
        win_log = []
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            # save model
            agent.save_model(exp_dir)
            print("Model saved")
        gc.collect()