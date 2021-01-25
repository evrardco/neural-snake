from math import isclose
from consts import *
from time import time
def is_opposite(alpha, beta):
    """
    returns -1 if opposite direction, otherwise 1
    """
    if isclose(abs(alpha - beta), 180.0):
        return -1
    return 0

def write_data(data, filename): 
    xs, ys = data
    with open(filename+f"{int(time()%100)}", "w+") as f:
        for i in range(len(xs)):
            for val in xs[i]:
                f.write(f"{val:.3f},")
            f.write(str(f"{ys[i][0]:.3f}"))
            f.write("\n")

def tuple_flatten_state(state):
    new_state = state.copy()
    new_state[BODY] = tuple(state[BODY])
    new_state[HUNGER] = -1
    return tuple(new_state.items())

