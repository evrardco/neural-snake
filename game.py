from consts import *
import game_logic

import data
import helper
import torch
import sys
import time
import rewards
import random
from os.path import join
import graphics
import keyboard
import pygame
from AI.nn_auto_encoder import NNAutoEncoder
from AI.lstm_ai import LSTMNN
clk = pygame.time.Clock()
from sklearn.preprocessing import MinMaxScaler
import numpy as np

fps = DEFAULT_FPS
new_model = lambda :  LSTMNN(
    12, 50, 3, 0.005, #model architecture
    data.data_simple,
    torch.device("cpu"), torch.float32, encoder_size=50) #others

prob_decay = 0.2
prob_rand = 1.0
num_rand = 10
games_per_gen = 30
games = num_rand
max_games = 100
def main():
    #rendering precomputing
    global fps, prob_decay, prob_rand, games
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background = pygame.Surface((WIDTH, HEIGHT))
    background.fill(graphics.black)
    dirty_rects = None
    explored_states = {}
    #ML precomputing
    data_x, data_y = [], []
    model = new_model()
    samples = 1000
    max_score = 0
    for i in range(10000):
        while games > 0:
            print(f"iteration {games_per_gen - games}")
            prob_rand = 0.1
            xs, ys = [], []
            state = game_logic.get_new_state()
            t0 = time.time()
            dt = 1
            old_score = 0
            misses = 0
            found = 0
            while state[ALIVE] and state[HUNGER] < MAX_HUNGER:
                
                new_x, new_y = model.recorder(state)
                xs.append(new_x)
                ys.append(new_y)


                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        #data.write_data((data_x, data_y), join("data", "dataset"))
                        sys.exit()

                dirty_rects = graphics.render_state(state, background, new_y[0], to_refresh=dirty_rects)


                state = game_logic.get_next_state(state, {TURN:model.pick_move(state, p_rand=prob_rand)})
                
                if state[SCORE] > old_score:
                    old_score = state[SCORE]
                    print(f"Added positive reward: {rewards.POSITIVE_RESPONSE}")
                    xs, ys = data.retro_affect(xs, ys, rewards.POSITIVE_RESPONSE, horizon=30)
                    print(ys[len(ys) - 20:])


                screen.blit(background, (0, 0))
                if dt > 0:
                    screen.blit(
                        graphics.font.render(f"fps: {int(1/dt)}, target:{fps}, data_len:{len(data_x)}", False, graphics.green),
                        FPS_OFFSETS
                    )
                pygame.display.update()
                clk.tick(fps)
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                fps = keyboard.update_fps(fps)
            old_max_score = max_score
            max_score = max(max_score, state[SCORE])
            if state[HUNGER] < MAX_HUNGER:
                xs, ys = data.retro_affect(xs, ys, rewards.NEGATIVE_RESPONSE, horizon=50)
            # if state[SCORE] >= 0.5 * max_score:
            for i in range(len(xs)):
                if ys[i][0] != 0:
                    print(f"{ys[i][0]=}")
                    data_x.append(xs[i])
                    data_y.append(ys[i])
                    data_x = data_x[max(len(data_x) - 5000, 0):]
                    data_y = data_y[max(len(data_y) - 5000, 0):]

            games -= 1
        prob_rand *= prob_decay
        model = new_model()
        if len(data_x) > 0:
            model.learn((data_x, data_y), 200)
        games = games_per_gen


        

        


if __name__ == "__main__":
    while True:
        main()