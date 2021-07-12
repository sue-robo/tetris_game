#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import pprint
import copy
from functools import reduce

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

def _get_board(GameStatus):
    height = GameStatus["field_info"]["height"]
    width = GameStatus["field_info"]["width"]
    board = np.array(GameStatus["field_info"]["backboard"]).reshape([height,width])
    board = np.where(board != 0, 1, 0)
    return board

def _get_board_features(board):
    features = {}
    peaks = _get_peaks(board)
    highest_peak = np.max(peaks)
    aggregated_height = np.array(peaks).sum()
    average_height = aggregated_height / board.shape[1]
    n_high_peaks = reduce(lambda r, x: r + 1 if x > 4 else 0, map(lambda x: x-average_height,peaks))

    holes = _get_holes(peaks, board)
    n_holes = np.sum(holes)
    n_col_with_holes = np.count_nonzero(np.array(holes)>0)
    row_transitions = _get_row_transitions(board, highest_peak)
    col_transitions = _get_col_transitions(board, peaks)
    bumpiness = _get_bumpiness(peaks)
    n_pits = np.count_nonzero(np.count_nonzero(board, axis=0) == 0)
    wells = _get_wells(peaks)
    max_wells = np.max(wells)
    max_wells_inner = np.max(wells[:len(wells)-1])

    features["highest_peak"] = highest_peak
    features["aggregated_height"] = aggregated_height
    features["average_height"] = average_height
    features["n_high_peaks"] = n_high_peaks
    features["n_holes"] = n_holes
    features["n_col_with_holes"] = n_col_with_holes
    features["row_transitions"] = row_transitions
    features["col_transitions"] = col_transitions
    features["bumpiness"] = bumpiness
    features["n_pits"] = n_pits
    features["max_wells"] = max_wells
    features["max_wells_inner"] = max_wells_inner
    return features

def _get_full_lines(board):
    res = [0, 0, 0, 0, 0]
    c = 0
    for h in range(0,len(board)):
        if all([ x != 0 for x in board[h,:]]):
            c += 1
        else:
            res[c] += 1
            c = 0
    return res[1:]

class Block_Controller_Manual():
    def __init__(self):
        self.ShapeNone_index = None
        self.board_width = 0

    def get_act(self, GameStatus):
        backboard = GameStatus["field_info"]["backboard"]
        self.board_height = GameStatus["field_info"]["height"]
        self.board_width = GameStatus["field_info"]["width"]
        CurrentShapeClass = GameStatus["block_info"]["currentShape"]["class"]
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]

        strategy = None
        LatestEvalValue = -100000
        for direction0 in CurrentShapeDirectionRange:
            x0Min, x0Max = self.getSearchXRange(CurrentShapeClass, direction0, self.board_width)
            for x0 in range(x0Min, x0Max):
                board = self.getBoard(backboard, CurrentShapeClass, direction0, x0)
                EvalValue = self.calcEvaluationValueSample(board)
                if EvalValue > LatestEvalValue:
                    strategy = {'direction':direction0, 'x':x0}
                    LatestEvalValue = EvalValue
        return strategy

    def getSearchXRange(self, ShapeClass, direction, board_width):
        #
        # get x range from shape direction.
        #
        minX, maxX, _, _ = ShapeClass.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
        xMin = -1 * minX
        xMax = board_width - maxX
        return xMin, xMax

    def getShapeCoordArray(self, ShapeClass, direction, x, y):
        #
        # get coordinate array by given shape.
        #
        coordArray = ShapeClass.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    def getBoard(self, backboard, Shape_class, direction, x):
        # 
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(backboard)
        _board = self.dropDown(board, Shape_class, direction, x)
        return _board

    def dropDown(self, board, Shape_class, direction, x):
        # 
        # internal function of getBoard.
        # -- drop down the shape on the board.
        # 
        dy = self.board_height - 1
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # update dy
        for _x, _y in coordArray:
            _yy = 0
            while _yy + _y < self.board_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_width + _x] == self.ShapeNone_index):
                _yy += 1
            _yy -= 1
            if _yy < dy:
                dy = _yy
        # get new board
        _board = self.dropDownWithDy(board, Shape_class, direction, x, dy)
        return _board

    def dropDownWithDy(self, board, Shape_class, direction, x, dy):
        #
        # internal function of dropDown.
        #
        _board = board
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        for _x, _y in coordArray:
            _board[(_y + dy) * self.board_width + _x] = Shape_class.shape
        return _board

    def calcEvaluationValueSample(self, board):
        flbonus = [10.0, 20.0, 50.0, 120.0]

        height = self.board_height
        width = self.board_width
        board = np.array(board).reshape((height,width))
        board = np.where(board != 0, 1, 0)
        features = _get_board_features(board)
        fullLines = _get_full_lines(board)
        fl = 0.0
        for b, l in zip(flbonus, fullLines):
            fl += b * l
        score = 0.0
        score = score + 1.0 * fl
        score = score - 5.0 * features["n_high_peaks"]
        score = score - 7.0 * features["n_holes"]
        score = score - 10.0 * features["n_col_with_holes"]
        score = score - 0.5 * features["bumpiness"]
        score = score - 1.0 * features["row_transitions"]
        score = score - 2.0 * (features["col_transitions"] - 10)
        score = score - 5.0 if features["max_wells_inner"] >= 3 else score
        score = score - 10.0 if features["average_height"] > 14 else score
        score = score - 10.0 if features["highest_peak"] > 18 else score
        return score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def get_state(GameStatus):
    height = GameStatus["field_info"]["height"]
    width = GameStatus["field_info"]["width"]
    block = GameStatus["block_info"]
    board = np.array(GameStatus["field_info"]["backboard"]).reshape([height,width])
    board = np.where(board != 0, 1, 0)
    def get_block(index, n):
        if index == 1:
            p = [0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0]
        elif index == 2:
            p = [0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif index == 3:
            p = [0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, n, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif index == 4:
            p = [0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, 0, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif index == 5:
            p = [0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif index == 6:
            p = [0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, n, n, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif index == 7:
            p = [0, 0, 0, 0, 0, n, n, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, n, n, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return np.array(p).reshape(4, width)

    cur_block = get_block(block["currentShape"]["index"], 2)
    nxt_block = get_block(block["nextShape"]["index"], 3)
    state = np.vstack((nxt_block, cur_block, board))
    return state

class DeepQNetwork(nn.Module):
    def __init__(self, output):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output)
        )
        #print(self.net)
    
    def forward(self, x):
        return self.net(x)


def select_reasonable_action(action_values, num_x, num_direction, xrange_tab):
    av = action_values.squeeze().detach().numpy()
    av = av.reshape([num_x, num_direction])
    mv, md, mx = -np.Inf, -1, -1
    for d in range(0, num_direction):
        xmin, xmax = xrange_tab[d]
        for x in range(xmin, xmax):
            v = av[x, d]
            if mv < v:
                mv, md, mx = v, d, x
    return mx * num_direction + md # action key

class DeepQNetworkAgent():
    def __init__(self, lr=1.0e-2, writer=None):
        self.num_direction = 4
        self.num_x = 10
        self.num_actions = self.num_x * self.num_direction # range(x) * range(direction)
        self.online_model = DeepQNetwork(self.num_actions)
        self.target_model = DeepQNetwork(self.num_actions)
        
        ''' if use existing network model, activate following two lines
        self.online_model.load_state_dict(torch.load('./dqn.prm'))
        self.target_model.load_state_dict(torch.load('./dqn_teacher.prm'))
        '''
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()

        self.max_experiences = 3000
        self.min_experiences = 300
        self.experience = {'s':[], 'a':[], 'r':[], 'n_s':[], 'done':[]}
        self.batch_size = 64

        self.optimizer = AdamW(self.online_model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.scheduler1 = ExponentialLR(self.optimizer, gamma=0.99995)

        self.epoch = 0
        self.writer = writer
        self.n_strategy = 0
        self.n_random = 0
        self.n_network = 0

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
        return (rewards + (1.0-dones) * gamma * next_Q)

    def act(self, state, epsilon, zehta=0.0, reference_strategy=None, xrange_tab=None):
        if np.random.rand() < epsilon:
            if np.random.rand() < zehta:
                direction = reference_strategy['direction']
                x = reference_strategy['x']
                self.n_strategy += 1
            else:
                direction = np.random.randint(0, self.num_direction)
                x = np.random.randint(0, self.num_x)
                self.n_random += 1
            action = x * self.num_direction + direction
            self.use_net = False
        else:
            self.online_model.eval()
            action_values = self.estimate(state)
            if xrange_tab is None:
                action = torch.argmax(action_values, axis=1).item()
            else:
                action = select_reasonable_action(action_values, self.num_x, self.num_direction, xrange_tab)
            self.use_net = True
            self.n_network += 1
        tot = self.n_strategy + self.n_random + self.n_network
        self.writer.flush()
        self.writer.add_scalars("policy rate",
                                {
                                    'strategy' : self.n_strategy/tot,
                                    'random'   : self.n_random/tot,
                                    'network'  : self.n_network/tot
                                },
                                tot)
        return action
    
    def lr_step(self):
        if self.epoch > 0 :
            self.scheduler1.step()

    def update_online(self, gamma):
        if len(self.experience['s']) < self.min_experiences:
            return
        states, next_states, actions, rewards, dones = self.recall()
        self.online_model.train()
        td_est = self.td_estimate(states, actions)
        td_tgt = self.td_target(next_states, rewards, dones, gamma)
        loss = self.criterion(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_net:
            self.writer.flush()
            self.writer.add_scalar("Train/loss", loss.item(), self.epoch)
            self.writer.add_scalars("learning rate",
                                    {
                                        'sched1'  : self.scheduler1.get_last_lr()[0],
                                        'optimlr' : self.optimizer.param_groups[0]['lr']
                                    },
                                    self.epoch)
            self.epoch += 1
    
    def update_target(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save_network(self, online_model_path, target_model_path):
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
    nextMove = {"strategy":
                    {
                        "direction": direction,
                        "x": x,
                        "y_operation": y_operation,
                        "y_moveblocknum": y_moveblocknum,
                    },
                }
    return nextMove

class DeepQNetworkTrainer():
    def __init__(self):
        self.gamma = 0.6
        self.epsilon = 1.00
        self.min_epsilon = 0.05
        self.epsilon_decay_rate = 0.9999
        self.zehta = 0.99
        self.min_zehta = 0.5
        self.zehta_decay_rate = 0.99995

        self.TARGET_UPDATE = 10

        self.agent = None
        self.reward_log = []
        self.prev_line_score_stat = [0, 0, 0, 0]
        self.prev_gameover_count = 0
        self.block_controller_sample = Block_Controller_Manual()

        self.iter = 0
        self.writer = SummaryWriter()

        self.ALMOST4LINES_BONUS = 20.0
        self.GAMEOVER_PENALTY = 500.0
        self.GAMECOMPLETE_BONUS = 100.0
        self.LINE_REWARDS = [10.0, 20.0, 50.0, 120.0]

    def _custom_reward(self, GameStatus, done, inner_iter):
        reward = 0.0
        line_score_stat = GameStatus["debug_info"]["line_score_stat"]
        gameover_count = GameStatus["judge_info"]["gameover_count"]
        board = _get_board(GameStatus)
        features = _get_board_features(board)
        line_reward = 0
        z = zip(self.LINE_REWARDS, line_score_stat, self.prev_line_score_stat)
        for i, (lr, ls, pls) in enumerate(z):
            line_reward += lr * (ls - pls)
            self.prev_line_score_stat[i] = line_score_stat[i]

        reward += 0.1 * inner_iter
        reward += line_reward
        reward -= 5.0 * features["n_high_peaks"]
        reward -= 7.0 * features["n_holes"]
        reward -= 10.0 * features["n_col_with_holes"]
        reward -= 0.5 * features["bumpiness"]
        reward -= 1.0 * features["row_transitions"]
        reward -= 2.0 * (features["col_transitions"] - 10)
        reward = reward - 5.0 if features["max_wells_inner"] >= 3 else reward
        reward = reward - 10.0 if features["average_height"] > 14 else reward
        reward = reward - 10.0 if features["highest_peak"] > 18 else reward

        reward = reward - 200.0 if not GameStatus["debug_info"]["r_operation"] else reward
        reward = reward - 200.0 if not GameStatus["debug_info"]["x_operation"] else reward

        def almost4lines():
            res = features["n_pits"] == 1
            res = res and features["max_wells"] >= 4
            res = res and features["average_height"] >= 4
            res = res and features["n_high_peaks"] == 0
            res = res and features["n_col_with_holes"] == 0
            return res

        if almost4lines():
            reward += self.ALMOST4LINES_BONUS
        if done:
            reward -= self.GAMEOVER_PENALTY
            self.prev_gameover_count = gameover_count
        if inner_iter >= 178:
            reward += self.GAMECOMPLETE_BONUS

        self.writer.flush()
        self.writer.add_scalar("Train/reward", reward, self.iter)
        self.writer.add_scalar("features/n_high_peaks", features["n_high_peaks"], self.iter)
        self.writer.add_scalar("features/n_holes", features["n_holes"], self.iter)
        self.writer.add_scalar("features/bumpiness", features["bumpiness"], self.iter)

        return reward

    def train(self, env, episode_cnt=1000):
        self.agent = DeepQNetworkAgent(lr=1.0e-2, writer=self.writer)
        self.iter = 0
        for _ in tqdm(range(episode_cnt)):
            gameStatus = env.reset()
            state = get_state(gameStatus)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)
            self.zehta = max(self.min_zehta, self.zehta * self.zehta_decay_rate)
            inner_iter, done = 0, False
            while not done and inner_iter < 180:
                self.iter, inner_iter = self.iter + 1, inner_iter + 1
                reference_strategy = self.block_controller_sample.get_act(gameStatus)
                cs = gameStatus["block_info"]["currentShape"]["class"]
                bw = gameStatus["field_info"]["width"]
                xrange_tab = []
                for d in range(0,4):
                    xrange_tab.append(self.block_controller_sample.getSearchXRange(cs, d, bw))
                action = self.agent.act(state, self.epsilon, self.zehta, reference_strategy)
                nextMove = get_next_move(action)
                gameStatus, reward, done = env.step(nextMove)
                prev_state, state = state, get_state(gameStatus)
                reward = self._custom_reward(gameStatus, done, inner_iter)
                exp = {'s':prev_state, 'a':action, 'r':reward, 'n_s':state, 'done':done}
                self.agent.add_experience(exp)
                self.agent.update_online(self.gamma)
                if self.iter % self.TARGET_UPDATE == 0:
                    self.agent.update_target()
            self.agent.lr_step()
            self.agent.save_network('dqn.prm', 'dqn_teacher.prm')

class Block_Controller(object):
    def __init__(self):
        self.num_direction = 4
        self.num_x = 10
        self.num_actions = self.num_x * self.num_direction
        self.model = DeepQNetwork(self.num_actions)
        #self.model.load_state_dict(torch.load('./game_manager/dqn.prm'))
        self.model.eval()

    def getSearchXRange(self, ShapeClass, direction, board_width):
        ''' The same as in the BlockControllerSample class
        '''
        minX, maxX, _, _ = ShapeClass.getBoundingOffsets(direction)
        xMin = -1 * minX
        xMax = board_width - maxX
        return xMin, xMax

    def GetNextMove(self, nextMove, GameStatus):
        t1 = datetime.now()

        # print GameStatus
        # print("=================================================>")
        # pprint.pprint(GameStatus, width = 61, compact = True)

        cs = GameStatus["block_info"]["currentShape"]["class"]
        bw = GameStatus["field_info"]["width"]
        xrange_tab = []
        for d in range(0,4):
            xrange_tab.append(self.getSearchXRange(cs,d,bw))

        state = get_state(GameStatus)

        action_values = self.model(
            torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(1)
        )
        action = select_reasonable_action(action_values, self.num_x, self.num_direction, xrange_tab)
        nextMove = get_next_move(action)

        # print("===", datetime.now() - t1)
        # print(nextMove)
        return nextMove

DQN_TRAINER = DeepQNetworkTrainer()
BLOCK_CONTROLLER = Block_Controller()
