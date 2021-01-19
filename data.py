from consts import *
import torch
import numpy as np
import game_logic
import random 
from math import cos, sin, radians, isclose
from time import time
from numba import jit
from helper import is_opposite
from rewards import *
DTYPE = torch.float32
DEVICE = torch.device("cpu")
def isclose(a, b):
    return abs(a - b) < 0.0001
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
        for j in range(max(WIDTH, HEIGHT)):
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

def learn(data, epochs, model, optimizer, criterion):
    x, y = data
    x, y = np.array(x), np.array(y)
    x_train = torch.tensor(x, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y_train = torch.tensor(y, dtype=DTYPE, device=DEVICE, requires_grad=True)
    model.train()
    for epoch in range(epochs):    
        optimizer.zero_grad()    # Forward pass
        y_pred = model(x_train)    # Compute Loss
        loss = criterion(y_pred, y_train)
        if epoch % 50 == 0:
            print('Epoch {}/{}: train loss: {}'.format(epoch, loss.item(), epochs))    # Backward pass
        loss.backward()
        optimizer.step()
    model.eval()

def pick_move(state, model, p_rand=0.0, data_recorder=get_data):
    preds = []
    if random.random() < p_rand:
        return random.choice(DIRS)
    for dir in DIRS:
        s = game_logic.get_next_state(state, {TURN: dir})
        x, _ = data_recorder(s)
        x_tensor = torch.tensor(np.array(x), dtype=DTYPE, device=DEVICE, requires_grad=True)
        pred = model(x_tensor).cpu().detach().numpy()[0]
        preds.append(pred)
    return DIRS[np.argmax(preds)]

def get_model(in_size, hidden_size, layers, learning_rate, encoder_size=4):
    return torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(in_size, hidden_size),
                    torch.nn.ReLU(),
                    *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for i in range(int(layers/2))],
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, encoder_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_size, hidden_size),
                    torch.nn.ReLU(),
                    *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for i in range(int(layers/2))],
                    torch.nn.Linear(hidden_size, 1)
                )

def data_to_sequences(xs, ys):
    return (
        [xs[i:i+3] for i in range(0, len(xs) - 3, 3)],
        [ys[i] for i in range(0, len(ys) - 3, 3)]
        )

class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

def get_rnn_model(in_size, hidden_size, layers, learning_rate):
    return torch.nn.Sequential(
                    RNNModel(in_size, hidden_size, 3, in_size),
                    torch.nn.Linear(in_size, hidden_size),
                    torch.nn.ReLU(),
                    *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for i in range(layers)],
                    torch.nn.Linear(hidden_size, 1)
                )



def learn_rnn(data, epochs, model, optimizer, criterion):
    x, y = data
    #y_2d_train = y
    #y_2d_train = torch.tensor(np.array(y_2d_train), dtype=DTYPE, device=DEVICE, requires_grad=True)
    x, y = data_to_sequences(x, y)
    x, y = np.array(x), np.array(y)
    x_train = torch.tensor(x, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y_train = torch.tensor(y, dtype=DTYPE, device=DEVICE, requires_grad=True)
    model.train()
    for epoch in range(epochs):    
        optimizer.zero_grad()    # Forward pass
        y_pred = model(x_train)    # Compute Loss
        loss = criterion(y_pred, y_train)
        if epoch % 50 == 0:
            print('Epoch {}/{}: train loss: {}'.format(epoch, loss.item(), epochs))    # Backward pass
        loss.backward()
        optimizer.step()
    model.eval()

def make_rnn_pred(state, history, model, p_rand=0.0, data_recorder=get_data_sensory):
    if len(history) < 2 or random.random() < p_rand:
        return random.choice(DIRS)
    inputs = history[len(history) - 2:]
    preds = []
    for dir in DIRS:

        s = game_logic.get_next_state(state, {TURN: dir})
        x, _ = data_recorder(s)
        x = [inputs + [x]]
        x_tensor = torch.tensor(np.array(x), dtype=DTYPE, device=DEVICE, requires_grad=True)
        pred = model(x_tensor).cpu().detach().numpy()[0]
        preds.append(pred)
    return DIRS[np.argmax(preds)]

def make_horizon(xs, ys, horizon=3):
    return (
        [xs[i] for i in range(0, len(xs), horizon)],
        [np.mean(ys[i:i+horizon]) for i in range(0, len(ys), horizon)]
        )
def retro_affect(xs, ys, reward, horizon=10):
    
    cutting_index = len(ys) - horizon
    if cutting_index <= 0:
        return (xs, ys)
    #print(ys[:cutting_index] + [[ys[cutting_index + i][0]/(horizon - i)] for i in range(0, horizon)] )
    return (
        xs,
        ys[:cutting_index] + [[ys[cutting_index + i][0] + (reward/(horizon - i))] for i in range(0, horizon)] 
    )
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
    return tuple(new_state.items())