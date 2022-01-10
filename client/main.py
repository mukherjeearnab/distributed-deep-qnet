# Import Libraries
import math
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from time import sleep

import relearn.pies.dqn as DQN
from relearn.explore import EXP, MEM
from relearn.pies.utils import compare_weights
from relearn.pies.utils import RMSprop_update

import modman

from queue import Queue
import gym


debug = False

now = datetime.datetime.now

# Logging CSV String
LOG_CSV = 'epoch,reward,tr,up\n'

##############################################
# SETUP Hyperparameters
##############################################
ALIAS = 'lunarlander_4x'
ENV_NAME = 'LunarLander-v2'

# API endpoint
URL = "http://localhost:5500/api/model/"


class INFRA:
    """ Dummy empty class"""

    def __init__(self):
        pass


EXP_PARAMS = INFRA()
EXP_PARAMS.MEM_CAP = 100000
EXP_PARAMS.EPST = (1.0, 0.01, 1.0)  # (start, min, max)
EXP_PARAMS.DECAY_MUL = 0.996
EXP_PARAMS.DECAY_ADD = 0


PIE_PARAMS = INFRA()
PIE_PARAMS.LAYERS = [128, 128, 128]
PIE_PARAMS.OPTIM = torch.optim.RMSprop
PIE_PARAMS.LOSS = torch.nn.MSELoss
PIE_PARAMS.LR = 0.001
PIE_PARAMS.DISCOUNT = 0.99
PIE_PARAMS.DOUBLE = False
PIE_PARAMS.TUF = 0
PIE_PARAMS.DEV = 'cpu'

TRAIN_PARAMS = INFRA()
TRAIN_PARAMS.EPOCHS = 500000
TRAIN_PARAMS.MOVES = 10
TRAIN_PARAMS.EPISODIC = False
TRAIN_PARAMS.BATCH_SIZE = 64
TRAIN_PARAMS.MIN_MEM = TRAIN_PARAMS.BATCH_SIZE*5
TRAIN_PARAMS.LEARN_STEPS = 1
TRAIN_PARAMS.TEST_FREQ = 100

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


##############################################
# Fetch Initial Model Params (If Available)
##############################################
global_params, n_push, is_available = modman.fetch_params(URL)

if is_available:
    P("Model exist")
    P("Loading Q params .....")
    pie.Q.load_state_dict(modman.convert_list_to_tensor(global_params))
    pie.Q.eval()
    P("Loading T params .....")
    pie.T.load_state_dict(pie.Q.state_dict())
    pie.T.eval()
else:
    P("Setting model for server")
    reply = modman.send_model_params(
        URL, modman.convert_tensor_to_list(pie.Q.state_dict()), PIE_PARAMS.LR)
    print(reply)


##############################################
# Training
##############################################
P('#', 'Train')
P('Start Training...')
stamp = now()
eps = []
ref = []
# n_send = 1
n_fetch = 1
max_reward1 = Queue(maxsize=10)

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
            if((epoch+1) % n_push == 0):
                modman.send_trained_params(
                    URL, modman.convert_tensor_to_list(pie.Q.state_dict()), epoch+1)

                # Wait for Model Lock to get Released
                while modman.get_model_lock(URL):
                    print("Waiting for Model Lock Release.")
                    sleep(0.1)
                # Get Updated Model Params from Server

                global_params, _, is_available = modman.fetch_params(URL)
                if debug:
                    P(".... Fetched model Params .....")
                if is_available:
                    if debug:
                        print("Loading gloabl parama ....")
                    pie.Q.load_state_dict(
                        modman.convert_list_to_tensor(global_params))
                    pie.Q.eval()
                else:
                    P("Model not avalable: Error!")

    # P("after explore epoch#:",epoch)
    if epoch == 0 or (epoch+1) % TRAIN_PARAMS.TEST_FREQ == 0:
        txp.reset(clear_mem=True)
        timesteps = txp.explore(
            pie, moves=1, decay=EXP.NO_DECAY, episodic=True)
        res = txp.summary(P=lambda *arg: None)
        trew = res[-1]
        ref.append([trew])

        if(max_reward1.full()):
            max_reward1.get()
        max_reward1.put(trew)

        P('[#]'+str(epoch+1), '\t',
            '[REW]'+str(trew),
            '[TR]'+str(pie.train_count),
            '[UP]'+str(pie.update_count))

        LOG_CSV += f'{str(epoch+1)},{str(trew)},{str(pie.train_count)},{str(pie.update_count)}\n'

        if(max_reward1.full()):
            if(np.mean(max_reward1.queue) >= 195):
                break

P('Finished Training!')
elapse = now() - stamp
P('Time Elapsed:', elapse)

##############################################
# Save Model And Training Log
##############################################
save_instance_path = f'./logs/{ENV_NAME}_{stamp.strftime("%d_%m_%Y-%H_%M_%S")}'

# Save Model
pie.save(save_instance_path + '.pt')

# Save Training Log
with open(save_instance_path + '.csv', 'w') as f:
    f.write(LOG_CSV)

modman.send_completed_model(
    URL, modman.convert_tensor_to_list(pie.Q.state_dict()))
