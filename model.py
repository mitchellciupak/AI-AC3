# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:47:22 2020

@author: mjcre
"""

#Import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Init and Set Variance of Tensor Weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out