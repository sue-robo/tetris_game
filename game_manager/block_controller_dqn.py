#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import pprint
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class DeepQNetwork(nn.Module):
    def __init__(self, input, output):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(64, output)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.out(x)
        return x

class DeepQNetworkAgent():
    def __init__(self, lr=1e-2):
        self.num_states = 11
        self.num_actions = 9*4 # range(x) * range(direction)
        self.model = DeepQNetwork(self.num_states, self.num_actions)
        self.teacher_model = DeepQNetwork(self.num_states, self.num_actions)

        self.max_experiences = 10_000
        self.min_experiences = 100
        self.experience = {'s':[], 'a':[], 'r':[], 'n_s':[], 'done':[]}
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.epoch = 0
        self.writer = SummaryWriter()

    def _get_peaks(self, board):
        res = np.array([])
        for c in range(board.shape[1]):
            if 1 in board[:, c]:
                p = board.shape[0] - np.argmax(board[:, c], axis=0)
                res = np.append(res, p)
            else:
                res = np.append(res, 0)
        return res

    def _get_holes(self, peaks, board):
        holes = []
        for c in range(board.shape[1]):
            start = -peaks[c]
            if start == 0:
                holes.append(0)
            else:
                holes.append(np.count_nonzero(board[int(start):, c] == 0))
        return holes

    def _get_row_transitions(self, board, highest_peak):
        sum = 0
        for r in range(int(board.shape[0]-highest_peak)):
            for c in range(1, board.shape[1]):
                if board[r, c] != board[r, c-1]:
                    sum += 1
        return sum

    def _get_col_transitions(self, board, peaks):
        sum = 0
        for c in range(board.shape[1]):
            if peaks[c] <= 1:
                continue
            for r in range(int(board.shape[0] - peaks[c]), board.shape[0]-1):
                if board[r,c] != board[r+1,c]:
                    sum += 1
        return sum

    def _get_bumpiness(self, peaks):
        sum = 0
        for i in range(len(peaks)-1):
            sum += np.abs(peaks[i]-peaks[i+1])
        return sum

    def _get_wells(self, peaks):
        wells = []
        for i in range(len(peaks)):
            if i == 0:
                w = peaks[1] - peaks[0]
                w = w if w > 0 else 0
                wells.append(w)
            elif i == len(peaks)-1:
                w = peaks[-2] - peaks[-1]
                w = w if w > 0 else 0
                wells.append(w)
            else:
                w1 = peaks[i-1] - peaks[i]
                w2 = peaks[i+1] - peaks[i]
                w1 = w1 if w1 > 0 else 0
                w2 = w2 if w2 > 0 else 0
                w = w1 if w1 >= w2 else w2
                wells.append(w)
        return wells

    def get_state(self, board, block):
        peaks = self._get_peaks(board)
        highest_peak = np.max(peaks)
        aggregated_height = np.sum(peaks)
        holes = self._get_holes(peaks, board)
        n_holes = np.sum(holes)
        n_col_with_holes = np.count_nonzero(np.array(holes)>0)
        row_transitions = self._get_row_transitions(board, highest_peak)
        col_transitions = self._get_col_transitions(board, peaks)
        bumpiness = self._get_bumpiness(peaks)
        n_pits = np.count_nonzero(np.count_nonzero(board, axis=0) == 0)
        wells = self._get_wells(peaks)
        max_wells = np.max(wells)
        current_shape = block["currentShape"]["index"]
        next_shape = block["nextShape"]["index"]
        return \
            highest_peak, \
            aggregated_height, \
            n_holes, \
            n_col_with_holes, \
            row_transitions, \
            col_transitions, \
            bumpiness, \
            n_pits, \
            max_wells, \
            current_shape, \
            next_shape

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def estimate(self, state):
        return self.model(torch.from_numpy(state).float())

    def future(self, state):
        return self.teacher_model(torch.from_numpy(state).float())

    def policy(self, state, epsilon):
        if random.random() < epsilon:
            direction = random.randint(0,4)
            x = random.randint(0,9)
            action = x * 4 + direction
        else:
            prediction = self.estimate(np.array(state)).detach().numpy()
            action = np.argmax(prediction)
        return action
    
    def get_next_move(self, action):
        direction = action % 4
        x = action // 4
        y_operation = 1
        y_moveblocknum = random.randint(1,8)
        return direction, x, y_operation, y_moveblocknum

    def update(self, gamma):
        if len(self.experience['s']) < self.min_experiences:
            return
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=32)
        states = np.asarray([self.experience['s'][i] for i in ids])
        states_next = np.asarray([self.experience['n_s'][i] for i in ids])

        print(f'shape of state = {states.shape}')

        estimateds = self.estimate(states).detach().numpy()
        future = self.future(states_next).detach().numpy()
        for idx, i in enumerate(ids):
            reward = self.experience['r'][i]
            done = self.experience['done'][i]
            if not done:
                future[idx] = reward + gamma * future[idx]
        loss = self.criterion(torch.tensor(estimateds, requires_grad=True),
                              torch.tensor(future, requires_grad=True))
        self.writer.add_scalar("Loss/train", loss, self.epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epoch += 1

    def update_teacher(self):
        self.teacher_model.load_state_dict(self.model.state_dict())

    def save_network(self, model_path):
        self.writer.flush()
        torch.save(self.model.state_dict(), model_path)

class Block_Controller(object):
    def __init__(self):
        print("Block_Controller::__init__()")
        self.agent = DeepQNetworkAgent()
        self.epsilon = 0.90
        self.epsilon_decay_rate = 0.999
        self.min_epsilon = 0.1
        self.prev_state = None
        self.prev_action = None
        self.prev_gameover_count = 0
        self.prev_score = 0
        self.gamma = 0.6


    def _custom_reward(self, reward, done):
        if done:
            if reward > 0:
                return 1
            elif reward < 0:
                return -1
            else:
                return 0
        else:
            return 0

    def callback_on_exit(self):
        self.agent.save_network('dqn.prm')

    def GetNextMove(self, nextMove, GameStatus):
        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        ##### add 
        height = GameStatus["field_info"]["height"]
        width = GameStatus["field_info"]["width"]
        board = GameStatus["field_info"]["backboard"]
        board = np.array(board).reshape([height, width])
        board = np.where(board != 0, 1, 0)
        block = GameStatus["block_info"]
        state = list(self.agent.get_state(board, block))
        pprint.pprint(state)
        score = GameStatus["judge_info"]["score"]
        done = GameStatus["judge_info"]["gameover_count"] != self.prev_gameover_count
        iter = GameStatus["judge_info"]["block_index"]
        reward = self._custom_reward(score - self.prev_score, done)
        self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay_rate)
        action = self.agent.policy(state, self.epsilon)

        if self.prev_state is not None:
            exp = {'s':self.prev_state, 'a':self.prev_action, 'r':reward, 'n_s':state, 'done':done}
            self.agent.add_experience(exp)
        self.agent.update(self.gamma)
        if iter % 100 == 0:
            self.agent.update_teacher()
        self.prev_state = state
        self.prev_action = action
        self.prev_score = score

        # return nextMove
        direction, x, y_operation, y_moveblocknum = self.agent.get_next_move(action)
        nextMove["strategy"]["direction"] = direction
        nextMove["strategy"]["x"] = x
        nextMove["strategy"]["y_operation"] = y_operation
        nextMove["strategy"]["y_moveblocknum"] = y_moveblocknum

        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

BLOCK_CONTROLLER = Block_Controller()