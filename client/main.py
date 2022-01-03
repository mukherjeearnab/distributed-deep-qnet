# Import Libraries
import math
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

import relearn.pies.dqn as DQN
from relearn.explore import EXP, MEM
from relearn.pies.utils import compare_weights

import modman

from queue import Queue
import gym

from copy import deepcopy


now = datetime.datetime.now

##############################################
# SETUP Hyperparameters
##############################################
ALIAS = 'experiment_01'
ENV_NAME = 'CartPole-v0'

# For test locally -> ..
# API endpoint
#URL = "http://localhost:5500/api/model/"  # Un comment this line if you wanna test locally
# ..

# For test in the server and sepertade clients ...

ip_address = "172.16.26.15"  # server macine ip address
# API endpoint
URL = "http://"+ip_address+":5500/api/model/"

# ..

class INFRA:
    """ Dummy empty class"""

    def __init__(self):
        pass


EXP_PARAMS = INFRA()
EXP_PARAMS.MEM_CAP = 50000
EXP_PARAMS.EPST = (0.95, 0.05, 0.95)  # (start, min, max)
EXP_PARAMS.DECAY_MUL = 0.99999
EXP_PARAMS.DECAY_ADD = 0


PIE_PARAMS = INFRA()
PIE_PARAMS.LAYERS = [128, 128, 128]
PIE_PARAMS.OPTIM = torch.optim.Adam
PIE_PARAMS.LOSS = torch.nn.MSELoss
PIE_PARAMS.LR = 0.001
PIE_PARAMS.DISCOUNT = 0.999999
PIE_PARAMS.DOUBLE = False
PIE_PARAMS.TUF = 4
PIE_PARAMS.DEV = 'cpu'

TRAIN_PARAMS = INFRA()
TRAIN_PARAMS.EPOCHS = 50000
TRAIN_PARAMS.MOVES = 10
TRAIN_PARAMS.EPISODIC = False
TRAIN_PARAMS.MIN_MEM = 30
TRAIN_PARAMS.LEARN_STEPS = 1
TRAIN_PARAMS.BATCH_SIZE = 50
TRAIN_PARAMS.TEST_FREQ = 10

TEST_PARAMS = INFRA()
TEST_PARAMS.CERF = 100
TEST_PARAMS.RERF = 100


P = print


def F(fig, file_name): return plt.close()  # print('FIGURE ::',file_name)


def T(header, table): return print(header, '\n', table)


P('#', ALIAS)

##############################################
# Setup ENVS
##############################################

# Train ENV
env = gym.make(ENV_NAME)

# Test ENV
venv = gym.make(ENV_NAME)

# Policy and Exploration
exp = EXP(env=env, cap=EXP_PARAMS.MEM_CAP, epsilonT=EXP_PARAMS.EPST)

txp = EXP(env=venv, cap=math.inf, epsilonT=(0, 0, 0))


def decayF(epsilon, moves, isdone):
    global eps
    new_epsilon = epsilon*EXP_PARAMS.DECAY_MUL + \
        EXP_PARAMS.DECAY_ADD  # random.random()
    eps.append(new_epsilon)
    return new_epsilon


pie = DQN.PIE(
    env.observation_space.shape[0],
    LL=PIE_PARAMS.LAYERS,
    action_dim=env.action_space.n,
    device=PIE_PARAMS.DEV,
    opt=PIE_PARAMS.OPTIM,
    cost=PIE_PARAMS.LOSS,
    lr=PIE_PARAMS.LR,
    dis=PIE_PARAMS.DISCOUNT,
    mapper=lambda x: x,
    double=PIE_PARAMS.DOUBLE,
    tuf=PIE_PARAMS.TUF,
    seed=None)

target = DQN.PIE(
    env.observation_space.shape[0],
    LL=PIE_PARAMS.LAYERS,
    action_dim=env.action_space.n,
    device=PIE_PARAMS.DEV,
    opt=PIE_PARAMS.OPTIM,
    cost=PIE_PARAMS.LOSS,
    lr=PIE_PARAMS.LR,
    dis=PIE_PARAMS.DISCOUNT,
    mapper=lambda x: x,
    double=PIE_PARAMS.DOUBLE,
    tuf=PIE_PARAMS.TUF,
    seed=None)

n_steps=20 #number of steps to learn then send params

##############################################
# Fetch Initial Model Params (If Available)
##############################################
global_params, is_available = modman.fetch_params(URL + 'get')

if is_available:
    pie.Q.load_state_dict(modman.convert_list_to_tensor(global_params))
else:
    reply = modman.send_model_update(
        URL + 'set', modman.convert_tensor_to_list(pie.Q.state_dict()), exp.memory.count, n_steps,None)
    print(reply)

##############################################
# Training
##############################################
P('#', 'Train')
P('Start Training...')
stamp = now()
eps = []
ref = []

max_reward1 = Queue(maxsize=100)

P('after max_reward queue')
exp.reset(clear_mem=True, reset_epsilon=True)
txp.reset(clear_mem=True, reset_epsilon=True)


for epoch in range(0, TRAIN_PARAMS.EPOCHS):

    # exploration
    _ = exp.explore(pie, moves=TRAIN_PARAMS.MOVES,
                    decay=decayF, episodic=TRAIN_PARAMS.EPISODIC)

    if exp.memory.count > TRAIN_PARAMS.MIN_MEM:

        for _ in range(TRAIN_PARAMS.LEARN_STEPS):
            # Single Learning Step
            pie.learn(exp.memory, TRAIN_PARAMS.BATCH_SIZE)

            # Send Gradients to Server
            if (epoch+1)%n_steps==0:
                reply = modman.send_model_update(URL + 'post_params',
                 modman.convert_tensor_to_list(pie.Q.state_dict()),
                 exp.memory.count, n_steps,None)
                print(reply)
                # Get Updated Model Params from Server
                global_params, is_available = modman.fetch_params(URL + 'get')
                pie.Q.load_state_dict(modman.convert_list_to_tensor(global_params))

    # P("after explore epoch#:",epoch)
    if epoch == 0 or (epoch+1) % TRAIN_PARAMS.TEST_FREQ == 0:
        txp.reset(clear_mem=True)
        timesteps = txp.explore(
            pie, moves=1, decay=EXP.NO_DECAY, episodic=True)
        res = txp.summary(P=lambda *arg: None)
        trew = res[-1]
        ref.append([trew])
        #print('before queue')
        if(max_reward1.full()):
            max_reward1.get()
        max_reward1.put(trew)
        #print('after queue')
        P('[#]'+str(epoch+1), '\t',
            '[REW]'+str(trew),
            '[TR]'+str(pie.train_count),
            '[UP]'+str(pie.update_count))

        if(max_reward1.full()):
            if(np.mean(max_reward1.queue) >= 200):
                break

P('Finished Training!')
elapse = now() - stamp
P('Time Elapsed:', elapse)
