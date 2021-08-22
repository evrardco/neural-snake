from AI.base import BaseAI
import torch
import numpy as np
from consts import *
import random
import game_logic
from sklearn.preprocessing import MinMaxScaler
class NNAutoEncoder(BaseAI):
    def __init__(self, in_size, hidden_size, layers, learning_rate, data_recorder, device, dtype, encoder_size=4):
        self.model = torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(in_size, hidden_size),
                    torch.nn.ReLU(),
                    *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for i in range(layers)],
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, encoder_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_size, 1)
                )
        self.optimizer = torch.optim.Adagrad(self.model.parameters())
        self.criterion = torch.nn.L1Loss()
        self.device = device
        self.dtype = dtype
        self.recorder = data_recorder
        self.y_scaler = None
    def learn(self, data, epochs):
        x, y = data
        x, y = np.array(x), np.array(y)
        x = MinMaxScaler().fit_transform(x)
        self.y_scaler = MinMaxScaler()
        self.y_scaler.fit(y)
        y = self.y_scaler.transform(y)
        x_train = torch.tensor(x, dtype=self.dtype, device=self.device, requires_grad=True)
        y_train = torch.tensor(y, dtype=self.dtype, device=self.device, requires_grad=True)
        self.model.train()
        for epoch in range(epochs):    
            self.optimizer.zero_grad()    # Forward pass
            y_pred = self.model(x_train)    # Compute Loss
            loss = self.criterion(y_pred, y_train)
            if epoch % 50 == 0:
                print('Epoch {}/{}: train loss: {}'.format(epoch, loss.item(), epochs))    # Backward pass
            loss.backward()
            self.optimizer.step()
        self.model.eval()
    def pick_move(self, state, p_rand=0.0):
        preds = []
        if random.random() < p_rand:
            return random.choice(DIRS)
        for dir in DIRS:
            s = game_logic.get_next_state(state, {TURN: dir})
            x, _ = self.recorder(s)
            x_tensor = torch.tensor(np.array(x), dtype=self.dtype, device=self.device, requires_grad=True)
            pred = self.model(x_tensor).cpu().detach().numpy()
            if self.y_scaler:
                pred = self.y_scaler.inverse_transform([pred])
            preds.append(pred[0])
        return DIRS[np.argmax(preds)]
