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