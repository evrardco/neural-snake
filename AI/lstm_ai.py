from AI.base import BaseAI
import torch
import numpy as np
from consts import *
import random
import game_logic
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Linear, ReLU, Sequential, Module, LSTM, MSELoss
class RecurrentNet(Module):
    def __init__(self,  in_size, hidden_size, layers, sequence_size=5):
        super(RecurrentNet, self).__init__()
        self.rec_layer = LSTM(in_size, in_size, 1)
        self.linear_layers = Sequential(
                    Linear(in_size, hidden_size),
                    ReLU(),
                    *[Sequential(Linear(hidden_size, hidden_size), ReLU()) for i in range(layers)],
                    ReLU(),
                    Linear(hidden_size, 1)
                )
    def forward(self, x):
        fw, _ = self.rec_layer(x)
        return self.linear_layers(fw)



class LSTMNN(BaseAI):
    def __init__(self, in_size, hidden_size, layers, learning_rate, data_recorder, device, dtype, sequence_size=5):
        self.in_size = in_size
        self.model = RecurrentNet(in_size, hidden_size, layers, sequence_size=sequence_size)
        self.sequence_size = 5
        self.optimizer = torch.optim.Adagrad(self.model.parameters())
        self.criterion = MSELoss()
        self.device = device
        self.dtype = dtype
        self.recorder = data_recorder
        self.y_scaler = None
        self.curr_sequence = [[[0 for _ in range(in_size)]] for _ in range(sequence_size)]
    def preprocess(self, data):
        curr_seq_x = [[0 for _ in range(self.in_size)] for _ in range(self.sequence_size)]
        curr_seq_y = [[0 for _ in range(self.in_size)] for _ in range(self.sequence_size)]
        x, y = data
        all_seqs_x = []
        all_seqs_y = []
        for i in range(len(x)):
            curr_seq_x.append(x[i])
            curr_seq_y.append(y[i])
            curr_seq_x = curr_seq_x[1:]
            curr_seq_y = curr_seq_y[1:]
            all_seqs_x.append(curr_seq_x.copy())
            all_seqs_y.append(curr_seq_y.copy())
        return all_seqs_x, all_seqs_y
    def learn(self, data, epochs):
        x_batches, y_batches = self.preprocess(data)
        x_train = torch.tensor(x_batches, dtype=self.dtype, device=self.device, requires_grad=True)
        y_train = torch.tensor(x_batches, dtype=self.dtype, device=self.device, requires_grad=True)
        self.model.train()
        for epoch in range(epochs):    
            self.optimizer.zero_grad()    # Forward pass
            y_pred = self.model(x_train)    # Compute Loss
            loss = self.criterion(y_pred, y_train)
            if epoch % 10 == 0:
                print('Epoch {}/{}: train loss: {}'.format(epoch, loss.item(), epochs))    # Backward pass
            loss.backward()
            self.optimizer.step()
        self.model.eval()
    def pick_move(self, state, p_rand=0.0):
        preds = []
        x, _ = self.recorder(state)
        self.curr_sequence.append([x])
        self.curr_sequence = self.curr_sequence[1:]
        if random.random() < p_rand:
            return random.choice(DIRS)
        for dir in DIRS:
            possible_x, _ = self.recorder(game_logic.get_next_state(state, {TURN: dir}))
            possible_sequence = self.curr_sequence.copy()
            possible_sequence.append([possible_x])
            possible_sequence = possible_sequence[1:]
            x_tensor = torch.tensor(np.array(possible_sequence), dtype=self.dtype, device=self.device, requires_grad=True)
            res = self.model(x_tensor)
            pred = self.model(x_tensor).cpu().detach().numpy()
            if self.y_scaler:
                pred = self.y_scaler.inverse_transform([pred])
            preds.append(pred[0])
        return DIRS[np.argmax(preds)]
