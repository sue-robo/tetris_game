#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import pprint
import random

class Block_Controller(object):
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

    def _get_state(self, board):
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
        return \
            highest_peak, \
            aggregated_height, \
            n_holes, \
            n_col_with_holes, \
            row_transitions, \
            col_transitions, \
            bumpiness, \
            n_pits, \
            max_wells

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
        state = self._get_state(board)
        pprint.pprint(state)

        

        # search best nextMove -->
        # random sample
        nextMove["strategy"]["direction"] = random.randint(0,4)
        nextMove["strategy"]["x"] = random.randint(0,9)
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = random.randint(1,8)
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

BLOCK_CONTROLLER = Block_Controller()