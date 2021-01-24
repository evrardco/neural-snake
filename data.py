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
Main.include("julia/data_getters.jl")
Main.include("julia/helpers.jl")
J_DataGetters = Main.DataGetters
DTYPE = torch.float32
DEVICE = torch.device("cpu")

def get_data(state):
    xs = []
    goal = NEUTRAL_RESPONSE 
    hx, hy = state[HEAD]
    xs += [hx, hy]
    fx, fy = state[FOOD]
    xs += [fx, fy]
    dir = state[DIR]
    dist = abs(fx - hx) + abs(fy - hy)
    xs.append(dist)
    goal -= HUNGER_FACTOR * state[HUNGER]
    if not state[HAS_FOOD]:
        print("GOT FOOD !!!!!! :D")
        goal = POSITIVE_RESPONSE
    elif state[ALIVE] and dist > 0:
        goal += 0#NEUTRAL_RESPONSE * len(state[BODY]) / dist**2
    
    dist_wall_x = min(abs(hx - WIDTH), hx)
    dist_wall_y = min(abs(hy - HEIGHT), hy)
    xs += [dist_wall_x, dist_wall_y]
    min_wall_dist = min(dist_wall_x, dist_wall_y)
    if  0 < min_wall_dist < BORDER_THRESHOLD:
        goal += 0#(NEGATIVE_RESPONSE + state[HUNGER]) / min_wall_dist**3 
    xs.append(state[DIR])
    xs.append(len(state[BODY]))
    xs.append(state[HUNGER])
    # food_LOS =  [
    #                 i for i in range(max(WIDTH, HEIGHT)) 
    #                 if isclose(hx + i * cos(radians(dir)) * TILE_SIZE, fx) and
    #                 isclose(hy + i * sin(radians(dir) * TILE_SIZE), fy)
    #             ]
    # if len(food_LOS) > 0:
    #     goal += POSITIVE_RESPONSE /  (1 + food_LOS[0])
    if not state[ALIVE]:
        goal = NEGATIVE_RESPONSE 

    return (xs, [goal])
def get_data_sensory(state, border_look_ahead=1000):
    dir = state[DIR]
    fx, fy = state[FOOD]
    hx, hy = state[HEAD]
    body = state[BODY]
    data = []
    for alpha in range(0, 360, 90):
        sensed = [0.0, 0.0, 0.0]
        for j in range(max(WIDTH, HEIGHT)//TILE_SIZE):
            mul = 1.0
            if isclose(alpha, dir):
                mul = POSITIVE_RESPONSE * is_opposite(alpha, dir)
            x = hx + j * TILE_SIZE * cos(radians(alpha))
            y = hy + j * TILE_SIZE * cos(radians(alpha))
            if isclose(x, fx) and isclose(y, fy):
                sensed[0] = 1 / (1 + j + int(random.random() * 2))
            if j < border_look_ahead and not ((0 <= x < WIDTH) and (0 <= y < HEIGHT)):
                sensed[1] = 1 / (1 + j + int(random.random() * 2))
            if j > 0:
                body_collisions = [True for part in body if isclose(x, part[0]) and isclose(y, part[1])]
                if len(body_collisions) > 0:
                    sensed[2] = 1 / (1 + j + int(random.random() * 2))
        data += sensed
    data += [abs(fx - hx) + abs(fy - hy)]
    #data.append(hx - fx)
    #data.append(hy - fy)
    _, y = get_data(state)
    return (data, y) 




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
