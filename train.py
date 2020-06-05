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