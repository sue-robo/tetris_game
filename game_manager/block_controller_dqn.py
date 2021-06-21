#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

''' # change model
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

class DeepQNetwork(nn.Module):
    def __init__(self, input, output):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        )

    def forward(self, x):
        return self.net(x)
'''
def get_state(GameStatus):
    height = GameStatus["field_info"]["height"]
    width = GameStatus["field_info"]["width"]
    block = GameStatus["block_info"]
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

class DeepQNetwork(nn.Module):
    def __init__(self, output):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output)
        )
        print(self.net)

    
    def forward(self, x):
        return self.net(x)


class DeepQNetworkAgent():
    def __init__(self, lr=1e-2):
        #self.board_h = 22
        #self.board_w = 10
        #self.block_h = 4
        self.num_direction = 4
        self.num_x = 10
        self.num_actions = self.num_x * self.num_direction # range(x) * range(direction)
        ''' # change model
        self.online_model = DeepQNetwork(11, self.num_actions)
        self.target_model = DeepQNetwork(11, self.num_actions)
        '''
        self.online_model = DeepQNetwork(self.num_actions)
        self.target_model = DeepQNetwork(self.num_actions)
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.max_experiences = 1000
        self.min_experiences = 100
        self.experience = {'s':[], 'a':[], 'r':[], 'n_s':[], 'done':[]}
        self.batch_size = 32

        self.optimizer = optim.Adam(self.online_model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.epoch = 0
        self.writer = SummaryWriter()


    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def recall(self):
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        states_next = np.asarray([self.experience['n_s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        return states, states_next, actions, rewards, dones

    ''' # change model
    def estimate(self, state):
        return self.online_model(
            torch.from_numpy(state.astype(np.float32))
        ).unsqueeze(0)

    def td_estimate(self, states, actions):
        est = self.online_model(torch.from_numpy(states.astype(np.float32)))
        current_Q = est[np.arange(0, self.batch_size), actions]
        return current_Q

    @torch.no_grad()
    def td_target(self, next_states, rewards, dones, gamma):
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()
        next_states = torch.from_numpy(next_states.astype(np.float32))
        next_state_Q = self.online_model(next_states)
        best_actions = torch.argmax(next_state_Q, axis=1)
        tgt = self.target_model(next_states)
        next_Q = tgt[np.arange(0, self.batch_size), best_actions]
        return (rewards + (1-dones) * gamma * next_Q)

    '''
    def estimate(self, state):
        return self.online_model(
            torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(1)
        )

    def td_estimate(self, states, actions):
        est = self.online_model(torch.from_numpy(states.astype(np.float32)).unsqueeze(1))
        current_Q = est[np.arange(0, self.batch_size), actions]
        return current_Q

    @torch.no_grad()
    def td_target(self, next_states, rewards, dones, gamma):
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()
        next_states = torch.from_numpy(next_states.astype(np.float32)).unsqueeze(1)
        next_state_Q = self.online_model(next_states)
        best_actions = torch.argmax(next_state_Q, axis=1)
        tgt = self.target_model(next_states)
        next_Q = tgt[np.arange(0, self.batch_size), best_actions]
        return (rewards + (1-dones) * gamma * next_Q)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            direction = np.random.randint(0, self.num_direction)
            x = np.random.randint(0, self.num_x)
            action = x * self.num_direction + direction
        else:
            action_values = self.estimate(state)
            action = torch.argmax(action_values, axis=1).item()
        return action
    
    def update_online(self, gamma):
        if len(self.experience['s']) < self.min_experiences:
            return
        states, next_states, actions, rewards, dones = self.recall()
        td_est = self.td_estimate(states, actions)
        td_tgt = self.td_target(next_states, rewards, dones, gamma)
        loss = self.criterion(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("Loss/train", loss.item(), self.epoch)
        self.epoch += 1

    def update_target(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save_network(self, online_model_path, target_model_path):
        self.writer.flush()
        torch.save(self.online_model.state_dict(), online_model_path)
        torch.save(self.target_model.state_dict(), target_model_path)

    def load_network(self, online_model_path, target_model_path=None):
        self.online_model.load_state_dict(torch.load(online_model_path))
        if target_model_path is not None:
            self.target_model.load_state_dict(torch.load(target_model_path))

def get_next_move(action):
    direction = action % 4
    x = action // 4
    y_operation = 1
    y_moveblocknum = np.random.randint(1,8)
    return direction, x, y_operation, y_moveblocknum

class DeepQNetworkTrainer():
    def __init__(self):
        self.epsilon = 0.9
        self.agent = None
        self.reward_log = []
        self.prev_line_score_stat = [0, 0, 0, 0]
        self.prev_gameover_count = 0

    def _custom_reward(self, GameStatus, done):
        line_score_stat = GameStatus["debug_info"]["line_score_stat"]
        gameover_count = GameStatus["judge_info"]["gameover_count"]
        line_reward = 0
        for i in range(4):
            line_reward += (i+1)*(i+1) * (line_score_stat[i] - self.prev_line_score_stat[i])
            self.prev_line_score_stat[i] = line_score_stat[i]
        gameover_penalty = (gameover_count - self.prev_gameover_count) * 100
        self.prev_gameover_count = gameover_count

        if done:
            #print(f'line_reward , gameover_penalty = {line_reward}, {gameover_penalty}')
            return line_reward - gameover_penalty
        else:
            return line_reward

    def train(self, env, episode_cnt=1000, min_epsilon=0.1, epsilon_decay_rate=0.99, gamma=0.6):
        self.agent = DeepQNetworkAgent()
        iter = 0
        for episode in tqdm(range(episode_cnt)):
            GameStatus = env.reset()
            state = get_state(GameStatus)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay_rate)
            done = False
            while not done:
                action = self.agent.act(state, self.epsilon)
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
                self.agent.update_online(gamma)
                iter += 1
                if iter % 100 == 0:
                    self.agent.update_target()
                #self.reward_log.append(reward)
        self.agent.save_network('dqn.prm', 'dqn_teacher.prm')
                
class Block_Controller(object):
    def __init__(self):
        self.num_direction = 4
        self.num_x = 10
        self.num_actions = self.num_x * self.num_direction # range(x) * range(direction)
        ''' # change model
        self.model = DeepQNetwork(11, self.num_actions)
        '''
        self.model = DeepQNetwork(self.num_actions)
        self.epsilon = 0.00
        #self.model.load_state_dict(torch.load('dqn.prm'))

    def GetNextMove(self, nextMove, GameStatus):
        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        ##### add 
        state = get_state(GameStatus)
        
        action = np.argmax(self.model(
            torch.from_numpy(state).view(-1, 1, state.shape[1], state.shape[0]).float()
        ).detach().numpy())
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
