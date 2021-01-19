from consts import *
import game_logic
import graphics
import keyboard
import data
import pygame
import torch
import sys
import time
import rewards
import random
clk = pygame.time.Clock()
fps = DEFAULT_FPS
new_model = lambda : data.get_model(10, 30, 2, 0.0001, encoder_size=8)
data_getter = data.J_DataGetters.data_simple
#new_model = lambda : data.get_rnn_model(24, 30, 4, 0.01)
prob_decay = 0.2
prob_rand = 1.0

def main():
    #rendering precomputing
    global fps, prob_decay, prob_rand
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background = pygame.Surface((WIDTH, HEIGHT))
    background.fill(graphics.black)
    dirty_rects = None
    explored_states = {}
    #ML precomputing
    data_x, data_y = [], []
    model = new_model()
    samples = 1000
    for i in range(10000):

        for j in range(10):
            print(f"iteration {j}")
            prob_rand = 0.1
            xs, ys = [], []
            state = game_logic.get_new_state()
            t0 = time.time()
            dt = 1
            old_score = 0
            while state[ALIVE] and state[HUNGER] < MAX_HUNGER:
                
                new_x, new_y = data_getter(state)
                xs.append(new_x)
                ys.append(new_y)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        data.write_data((data_x, data_y), "dataset")
                        sys.exit()
                dirty_rects = graphics.render_state(state, background, new_y[0], to_refresh=dirty_rects)
                state = game_logic.get_next_state(state, {TURN:data.pick_move(state, model, p_rand=prob_rand, data_recorder=data_getter)})
                
                if state[SCORE] > old_score:
                    old_score = state[SCORE]
                    print(f"Added positive response: {rewards.POSITIVE_RESPONSE}")
                    xs, ys = data.retro_affect(xs, ys, rewards.POSITIVE_RESPONSE, horizon=20)
                    print(ys[len(ys) - 20:])
                screen.blit(background, (0, 0))
                if dt > 0:
                    screen.blit(
                        graphics.font.render(f"fps: {int(1/dt)}, target:{fps}, it:{i+j}, data_len:{len(data_x)}", False, graphics.green),
                        FPS_OFFSETS
                    )
                pygame.display.update()
                clk.tick(fps)
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                fps = keyboard.update_fps(fps)
            if state[HUNGER] < MAX_HUNGER:
                xs, ys = data.retro_affect(xs, ys, rewards.NEGATIVE_RESPONSE, horizon=20)
            frozen = data.tuple_flatten_state(state)
            if frozen not in explored_states:
                explored_states[frozen] = True
                data_x += xs
                data_y += ys                
        prob_rand *= prob_decay
        #model = new_model()
        data.learn((data_x, data_y), 500, model, torch.optim.Adagrad(model.parameters()),torch.nn.MSELoss())
        #data.learn_rnn((data_x, data_y), 100, model, torch.optim.Adagrad(model.parameters()),torch.nn.MSELoss())

        

        


if __name__ == "__main__":
    while True:
        main()