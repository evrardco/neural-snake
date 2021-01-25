from consts import *
import torch
import numpy as np
import game_logic
import random 
from math import cos, sin, radians, isclose
from time import time
from helper import is_opposite
from rewards import *
from julia import Main
#Import julia data getters
Main.include("julia/data_getters.jl")
Main.include("julia/helpers.jl")
J_DataGetters = Main.DataGetters
DTYPE = torch.float32
DEVICE = torch.device("cpu")
#Bind retro affect to its julia counterpart
def retro_affect(xs, ys, reward, horizon=10):
    """
    inputs:
        xs: a list of vectors of the predictor variables
        ys: a list of  1-sized vectors of the predicted variables
    outputs:
        tuple containing xs and ys such that the 'horizon' last elements of ys have been added
        a part of an additional rewards (based on the distance from the end of the list)
    """
    ret = Main.Helpers.retroAffect(xs, ys, reward, horizon)
    return ret
