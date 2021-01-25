from consts import *
from random import randint, random, choice
from math import sin, cos, radians, isclose

def get_food_pos():
    """
    Create a random food position based on the game parameters
    output: (x, y) the food position
    """
    return (
        randint(1, (WIDTH//TILE_SIZE) - 1) * TILE_SIZE,
        randint(1, (HEIGHT//TILE_SIZE) - 1) * TILE_SIZE
    )

def get_new_state():
    """
    Builds a new game state with everything initializated correctly.
    output: dict with keys as described in consts.py
    """
    snake_head = (
        randint(SNAKE_SIZE + 1, (WIDTH//TILE_SIZE) - SNAKE_SIZE) * TILE_SIZE,
        randint(SNAKE_SIZE + 1,  (HEIGHT//TILE_SIZE) - SNAKE_SIZE) * TILE_SIZE
    )
    snake_dir = choice(DIRS)
    snake_body = [
        (snake_head[0] - i * TILE_SIZE * cos(radians(snake_dir)), snake_head[1] - i * TILE_SIZE * sin(radians(snake_dir))) for i in range(SNAKE_SIZE)
    ]
    food_pos = get_food_pos()
    alive = True
    score = 0
    return {
        HEAD : snake_head,
        BODY : snake_body,
        DIR : snake_dir,
        FOOD : food_pos,
        HAS_FOOD : True,
        SCORE : score,
        ALIVE: alive,
        HUNGER: 0,
        HIST: [snake_dir for _ in range(HIST_SIZE)]
    }

"""
given the state at time t and some events,
computes the next state
inputs: 
    state: The state at time t
    events: Which changes occur
outputs:
    state at time t +
"""
def get_next_state(state, events):
    turn_dir = events[TURN]
    direction = state[DIR]
    body = state[BODY].copy()
    hx, hy = state[HEAD]
    fx, fy = state[FOOD]
    has_food = state[HAS_FOOD]
    score = state[SCORE]
    alive = state[ALIVE]
    hunger = state[HUNGER] + 1
    hist = state[HIST]

    if hunger >= MAX_HUNGER:
        alive = False
    #Make the snake go forward
    if turn_dir is not None and abs(direction - turn_dir) != 180:
        direction = turn_dir
    hx = hx + round(cos(radians(direction)) * TILE_SIZE)
    hy = hy + round(sin(radians(direction)) * TILE_SIZE)
    body = body[:len(body) - 1]

    #Check game over conditions
    if not ((0 <= hx < WIDTH) and (0 <= hy < HEIGHT)):
        alive = False
    else:
        for px, py in body:
            if isclose(px, hx) and isclose(py, hy):
                alive = False
    body.insert(0, (hx, hy))

    
    if has_food and isclose(fx, hx) and isclose(fy, hy):
        has_food = False
        body.insert(0, (hx, hy))
        score += 1
        hunger = 0
    elif not has_food:
        fx, fy = get_food_pos()
        has_food = True

    

    return {
        DIR: direction,
        BODY: body,
        HEAD : (hx, hy),
        FOOD : (fx, fy),
        HAS_FOOD: has_food,
        ALIVE: alive,
        SCORE: score,
        HUNGER: hunger,
        HIST: hist.copy()[len(hist) - HIST_SIZE:]
    }