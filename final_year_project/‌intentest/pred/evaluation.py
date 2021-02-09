from pathlib import Path

import numpy as np
from itertools import groupby
from constants.enum_keys import PG
from pgdataset.s0_label import PgdLabel
from torch.utils.data import DataLoader
from pred.play_gesture_results import Player


class Eval:
    def __init__(self):
        self.player = Player()
        self.num_video = len(PgdLabel(Path.home() / 'PoliceGestureLong', is_train=False))
        self.ed = EditDistance()

    def main(self):
        for n in range(self.num_video):
            res = self.player.play_dataset_video(is_train=False, video_index=n, show=False)
            target = res[PG.GESTURE_LABEL]
            source = res[PG.PRED_GESTURES]
            assert len(source) == len(target)
            source_group = [k for k, g in groupby(source)]
            target_group = [k for k, g in groupby(target)]
            S, D, I = self.ed.edit_distance(source_group, target_group)
            print('S:%d, D:%d, I:%d'%(S, D, I))
            pass


class EditDistance:
    def edit_distance(self, word1, word2):
        word2 = tuple(word2)
        return self.__edit_distance(word1, word2, {})

    def __edit_distance(self, word1, word2, computed_solutions):
        if len(word1) == 0:
            return 0, 0, len(word2)  
        if len(word2) == 0:
            return 0, len(word1), 0  

        replace_tuple = (word1[1:], word2[1:])
        delete_tuple = (word1[1:], word2)
        insert_tuple = (word1, word2[1:])

        replace_dist = self.__distance_add(self.__replace_cost(word1, word2), self.__transformation_cost(replace_tuple, computed_solutions))
        delete_dist = self.__distance_add((0, 1, 0), self.__transformation_cost(delete_tuple, computed_solutions))
        insert_dist = self.__distance_add((0, 0, 1), self.__transformation_cost(insert_tuple, computed_solutions))

        min_dist = self.__distance_min(replace_dist, delete_dist, insert_dist)
        return min_dist

    def __replace_cost(self, word1, word2):
        if word1[0] == word2[0]:
            return 0,0,0 
        else:
            return 1,0,0 

    def __transformation_cost(self, problem_tuple, solutions):
  
        if problem_tuple in solutions:  
            return solutions.get(problem_tuple)
        else:
            distSDI = self.__edit_distance(problem_tuple[0], problem_tuple[1], solutions) 
            solutions[problem_tuple] = distSDI
            return distSDI

    def __distance_add(self, dis1, dis2):


        S = dis1[0] + dis2[0]
        D = dis1[1] + dis2[1]
        I = dis1[2] + dis2[2]
        return (S,D,I)

    def __distance_min(self, dis1, dis2, dis3):
       

        d1_total = dis1[0] + dis1[1] + dis1[2]
        d2_total = dis2[0] + dis2[1] + dis2[2]
        d3_total = dis3[0] + dis3[1] + dis3[2]
        arr123 = np.array([d1_total, d2_total, d3_total], np.int32)
        argminimum = int(np.argmin(arr123))
        if argminimum == 0:
            return dis1
        elif argminimum == 1:
            return dis2
        elif argminimum == 2:
            return dis3
        else:
            raise ValueError()
