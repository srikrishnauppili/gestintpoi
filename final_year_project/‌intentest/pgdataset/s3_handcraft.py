import numpy as np
import math
from pathlib import Path
from pgdataset.s2_truncate import PgdTruncate
from constants.enum_keys import PG
from constants.keypoints import aic_bones, aic_bone_pairs


class PgdHandcraft(PgdTruncate):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(data_path, is_train, resize_img_size, clip_len)
        self.bla = BoneLengthAngle()

    def __getitem__(self, index):
        try:
            res_dict = super().__getitem__(index)
            feature_dict = self.bla.handcrafted_features(res_dict[PG.COORD_NORM])
            res_dict.update(feature_dict)
            return res_dict
        except:
            print("error")


class BoneLengthAngle:

    def __init__(self):
        self.connections = np.asarray(aic_bones, np.int) - 1
        self.pairs = np.asarray(aic_bone_pairs, np.int) - 1

    def handcrafted_features(self, coord_norm):
        assert len(coord_norm.shape) == 3  
        feature_dict = {}
        bone_len = self.__bone_len(coord_norm)
        bone_sin, bone_cos = self.__bone_pair_angle(coord_norm)
        feature_dict[PG.BONE_LENGTH] = bone_len
        feature_dict[PG.BONE_ANGLE_SIN] = bone_sin
        feature_dict[PG.BONE_ANGLE_COS] = bone_cos
        return feature_dict

    def __bone_len(self, coord):

        xy_coord = np.asarray(coord)  
        
        xy_val = np.take(xy_coord, self.connections, axis=2)
        xy_diff = xy_val[:, :, :, 0] - xy_val[:, :, :, 1]  
        xy_diff = xy_diff ** 2  
        bone_len = np.sqrt(xy_diff[:, 0] + xy_diff[:, 1])  

        return bone_len

    def __bone_pair_angle(self, coord):
        xy_coord = np.asarray(coord) 
        xy_val = np.take(xy_coord, self.pairs, axis=2) 
        xy_vec = xy_val[:, :, :, :, 1] - xy_val[:, :, :, :, 0]  
        ax = xy_vec[:, 0, :, 0]
        bx = xy_vec[:, 0, :, 1]
        ay = xy_vec[:, 1, :, 0]
        by = xy_vec[:, 1, :, 1]
        dot_product = ax * bx + ay * by  
        cross_product = ax * by - ay * bx  
        magnitude = np.einsum('fxpb,fxpb->fpb', xy_vec, xy_vec)  
        magnitude = np.sqrt(magnitude) 
        magnitude[magnitude < 10e-3] = 10e-3  
        mag_AxB = magnitude[:, :, 0] * magnitude[:, :, 1]  
        cos = dot_product / mag_AxB
        sin = cross_product / mag_AxB
        return sin, cos