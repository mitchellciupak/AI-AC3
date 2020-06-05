# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:32:21 2020

@author: mjcre
"""

# Import
import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

# Init ensurance the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
        
def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # shifting the seed with rank to asynchronize each training agent
    env = create_atari_env(params.env_name) # creating an optimized environment thanks to the create_atari_env function
    env.seed(params.seed + rank) # aligning the seed of the environment on the seed of the agent
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # creating the model from the ActorCritic class
    state = env.reset() # state is a numpy array of size 1*42*42, in black & white
    state = torch.from_numpy(state) # converting the numpy array into a torch tensor
    done = True # when the game is done
    episode_length = 0 # initializing the length of an episode to 0