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

from tqdm import tqdm


'''
def _get_peaks(board):
    res = np.array([])
    for c in range(board.shape[1]):
        if 1 in board[:, c]:
            p = board.shape[0] - np.argmax(board[:, c], axis=0)
            res = np.append(res, p)
        else:
            res = np.append(res, 0)
    return res

def _get_holes(peaks, board):
    holes = []
    for c in range(board.shape[1]):
        start = -peaks[c]
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(board[int(start):, c] == 0))
    return holes

def _get_row_transitions(board, highest_peak):
    sum = 0
    for r in range(int(board.shape[0]-highest_peak)):
        for c in range(1, board.shape[1]):
            if board[r, c] != board[r, c-1]:
                sum += 1
    return sum

def _get_col_transitions(board, peaks):
    sum = 0
    for c in range(board.shape[1]):
        if peaks[c] <= 1:
            continue
        for r in range(int(board.shape[0] - peaks[c]), board.shape[0]-1):
            if board[r,c] != board[r+1,c]:
                sum += 1
    return sum

def _get_bumpiness(peaks):
    sum = 0
    for i in range(len(peaks)-1):
        sum += np.abs(peaks[i]-peaks[i+1])
    return sum

def _get_wells(peaks):
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
def get_state(GameStatus):
    height = GameStatus["field_info"]["height"]
    width = GameStatus["field_info"]["width"]
    block = GameStatus["block_info"]
    board = np.array(GameStatus["field_info"]["backboard"]).reshape([height,width])
    board = np.where(board != 0, 1, 0)

    peaks = _get_peaks(board)
    highest_peak = np.max(peaks)
    aggregated_height = np.sum(peaks)
    holes = _get_holes(peaks, board)
    n_holes = np.sum(holes)
    n_col_with_holes = np.count_nonzero(np.array(holes)>0)
    row_transitions = _get_row_transitions(board, highest_peak)
    col_transitions = _get_col_transitions(board, peaks)
    bumpiness = _get_bumpiness(peaks)
    n_pits = np.count_nonzero(np.count_nonzero(board, axis=0) == 0)
    wells = _get_wells(peaks)
    max_wells = np.max(wells)
    current_shape = block["currentShape"]["index"]
    next_shape = block["nextShape"]["index"]
    return np.array([
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
                    ])
'''
def get_state(GameStatus):
    height = GameStatus["field_info"]["height"]
    width = GameStatus["field_info"]["width"]
    block = GameStatus["block_info"]
    '''
    board = np.array(GameStatus["field_info"]["backboard"]).reshape([height,width])
    '''
    board = np.array(GameStatus["field_info"]["withblock"]).reshape([height,width])
    board = np.where(board != 0, 1, 0)
    if block["nextShape"]["index"] == 1:
        state = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 2:
        state = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 3:
        state = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 2, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 4:
        state = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 5:
        state = [0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 6:
        state = [0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 2, 2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif block["nextShape"]["index"] == 7:
        state = [0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 2, 2, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    state = np.array(state).reshape(4, 10)
    state = np.vstack((state, board))
    return state

def get_next_move(action):
    direction = action % 4
    x = action // 4
    y_operation = 1
    y_moveblocknum = random.randint(1,8)
    return direction, x, y_operation, y_moveblocknum


'''
class DeepQNetwork(nn.Module):
    def __init__(self, input, output):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
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
'''
class DeepQNetwork(nn.Module):
    def __init__(self, output):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2560, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.head = nn.Linear(256, output)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x

class DeepQNetworkAgent():
    def __init__(self, lr=1e-2):
        self.board_h = 22
        self.board_w = 10
        self.block_h = 4
        self.num_states = self.board_h * self.board_w
        self.num_actions = 9*4 # range(x) * range(direction)
        self.model = DeepQNetwork(self.num_actions)
        self.teacher_model = DeepQNetwork(self.num_actions)

        self.max_experiences = 10_000
        self.min_experiences = 100
        self.experience = {'s':[], 'a':[], 'r':[], 'n_s':[], 'done':[]}
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.epoch = 0
        self.writer = SummaryWriter()


    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def estimate(self, state):
        return self.model(
            torch.from_numpy(state).view(-1, 1, self.board_w, self.board_h+self.block_h).float()
        )

    def future(self, state):
        return self.teacher_model(
            torch.from_numpy(state).view(-1, 1, self.board_w, self.board_h+self.block_h).float()
        )

    def policy(self, state, epsilon):
        if random.random() < epsilon:
            direction = random.randint(0,4)
            x = random.randint(0,9)
            action = x * 4 + direction
        else:
            prediction = self.estimate(state).detach().numpy()
            action = np.argmax(prediction)
        return action
    
    def update(self, gamma):
        if len(self.experience['s']) < self.min_experiences:
            return
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=32)
        states = np.asarray([self.experience['s'][i] for i in ids])
        states_next = np.asarray([self.experience['n_s'][i] for i in ids])
        estimateds = self.estimate(states).detach().numpy()
        future = self.future(states_next).detach().numpy()
        for idx, i in enumerate(ids):
            reward = self.experience['r'][i]
            done = self.experience['done'][i]
            if not done:
                future[idx] = reward + gamma * future[idx]
        loss = self.criterion(torch.tensor(estimateds, requires_grad=True),
                              torch.tensor(future, requires_grad=True))
        self.writer.add_scalar("Loss/train", loss.item(), self.epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epoch += 1

    def update_teacher(self):
        self.teacher_model.load_state_dict(self.model.state_dict())

    def save_network(self, model_path, teacher_model_path):
        self.writer.flush()
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.teacher_model.state_dict, teacher_model_path)

    def load_network(self, model_path, teacher_model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.teacher_model.load_state_dict(torch.load(teacher_model_path))

class DeepQNetworkTrainer():
    def __init__(self):
        self.epsilon = 0.9
        self.agent = None
        self.reward_log = []

    def _custom_reward(self, GameStatus, done):
        line_score_stat = GameStatus["debug_info"]["line_score_stat"]
        gameover_count = GameStatus["judge_info"]["gameover_count"]
        line_reward = 0
        for i in range(4):
            line_reward += (i+1)*(i+1) * line_score_stat[i]
        gameover_penalty = gameover_count * 5

        if done:
            return line_reward - gameover_penalty
        else:
            return 0

    def train(self, env, episode_cnt=1000, min_epsilon=0.1, epsilon_decay_rate=0.99, gamma=0.6):
        self.agent = DeepQNetworkAgent()
        iter = 0
        for episode in tqdm(range(episode_cnt)):
            GameStatus = env.reset()
            state = get_state(GameStatus)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay_rate)
            done = False
            while not done:
                action = self.agent.policy(state, self.epsilon)
                prev_state = state
                direction, x, y_operation, y_moveblocknum = get_next_move(action)
                nextMove = {"strategy":
                                {
                                    "direction": direction,
                                    "x": x,
                                    "y_operation": y_operation,
                                    "y_moveblocknum": y_moveblocknum,
                                },
                            }
                GameStatus, reward, done = env.step(nextMove)
                state = get_state(GameStatus)
                reward = self._custom_reward(GameStatus, done)
                exp = {'s':prev_state, 'a':action, 'r':reward, 'n_s':state, 'done':done}
                self.agent.add_experience(exp)
                self.agent.update(gamma)
                iter += 1
                if iter % 100 == 0:
                    self.agent.update_teacher()
                self.reward_log.append(reward)
        self.agent.save_network('dqn.prm')
                
class Block_Controller(object):
    def __init__(self):
        self.num_actions = 9*4
        self.model = DeepQNetwork(self.num_actions)
        self.epsilon = 0.00
        self.model.load_state_dict(torch.load('dqn.prm'))

    def GetNextMove(self, nextMove, GameStatus):
        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        ##### add 
        state = get_state(GameStatus)
        
        prediction = self.model(
            torch.from_numpy(state).view(-1, 1, state.shape[1], state.shape[0]).float()
        ).detach().numpy()
        action = np.argmax(prediction)
        direction, x, y_operation, y_moveblocknum = get_next_move(action)
        nextMove["strategy"]["direction"] = direction
        nextMove["strategy"]["x"] = x
        nextMove["strategy"]["y_operation"] = y_operation
        nextMove["strategy"]["y_moveblocknum"] = y_moveblocknum

        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

DQN_TRAINER = DeepQNetworkTrainer()
BLOCK_CONTROLLER = Block_Controller()
