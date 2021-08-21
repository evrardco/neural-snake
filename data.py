from consts import *
import torch
import numpy as np
import game_logic
import random 
from math import cos, sin, radians, isclose
from time import time
from helper import is_opposite
from rewards import *
# from julia import Main
# #Import julia data getters
# Main.include("julia/data_getters.jl")
# Main.include("julia/helpers.jl")
# J_DataGetters = Main.DataGetters
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

    horizon = min(len(ys), horizon)
    N = len(ys)
    for i in range(horizon):
        ys[(N - 1) - i][0] = ys[(N - 1) - i][0] + (reward / (i + 1)) 
    
    return (xs, ys)



POSITIVE_RESPONSE = 100.0
NEGATIVE_RESPONSE = -20.0
NEUTRAL_RESPONSE = 1.0
BORDER_THRESHOLD = 10.0
HUNGER_FACTOR = 0.01


def data_simple(state):
    xs = [1 if state[ALIVE] else 0]
    fx, fy = state[FOOD]
    dir = state[DIR]
    hx, hy = state[HEAD]
    fx, fy = state[FOOD]
    xs += [hx, hy]
    xs += [fx, fy]
    goal = 0.0
    dist = abs(fx - hx) + abs(fy - hy)
    xs.append(dist)
    if not state[HAS_FOOD]:
        goal = POSITIVE_RESPONSE * state[SCORE]
    
    dist_wall_x = min(abs(hx - WIDTH), hx)
    dist_wall_y = min(abs(hy - HEIGHT), hy)
    dist_wall_food_x = min(abs(fx - WIDTH), fx)
    dist_wall_food_y = min(abs(fy - HEIGHT), fy)
    xs += [dist_wall_x, dist_wall_y]
    xs += [dist_wall_food_x, dist_wall_food_y]

    xs.append(state[DIR])
    xs.append(len(state[BODY]))
    xs.append(state[HUNGER])
    return (xs, [goal])


def data_sensory(state, border_look_ahead=5):
    dir = state[DIR]
    fx, fy = state[FOOD]
    hx, hy = state[HEAD]
    body = state[BODY]
    data = []
    for alpha in range(int(dir - 90), int(dir + 90), 90):
        sensed = [0.0, 0.0, 0.0]
        for j in range(int(max(WIDTH, HEIGHT)/TILE_SIZE)):
            mul = 1.0
            x = hx + j * TILE_SIZE * cos(alpha)
            y = hy + j * TILE_SIZE * cos(alpha)
            if isclose(x, fx) and isclose(y, fy):
                sensed[0] = 1 / (j + 1)
            if ((0 <= x < WIDTH) and (0 <= y < HEIGHT)):
                sensed[1] = 1 / (j + 1)
            if j > 0:
                for part in body:
                    if isclose(x, part[0]) and isclose(y, part[1]):
                        sensed[2] = 1 / (j + 1)
        data += sensed
    #Computing how many neighbouring tiles from the head are occupied
    #by the snake's body.
    radius = 3
    for r in range(1, radius):
        closeness = 0
        for tx in range(hx - r * TILE_SIZE, hx + r * TILE_SIZE, TILE_SIZE):
            for ty in range(hy - r * TILE_SIZE, hy + r * TILE_SIZE, TILE_SIZE):
                for (px, py) in body[1:]: #2 to not take the head into account
                    if isclose(px, tx) and isclose(py, ty):
                        closeness += 1
        data.append(closeness)
    xs, y = data_simple(state)
    data += xs
    data += state[HIST]
    return (data, y)
