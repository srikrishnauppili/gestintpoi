from pathlib import Path

import cv2
import numpy as np
from imgaug import KeypointsOnImage

from constants.enum_keys import PG
from pgdataset.s1_skeleton import PgdSkeleton


class Player:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest
        self.img_size = (512, 512)

    def play(self, is_train, video_index):
        self.scd = PgdSkeleton(Path.home() / 'PoliceGestureLong', is_train, self.img_size)
        res = self.scd[video_index]
        coord_norm = res[PG.COORD_NORM]  
        coord_norm = np.transpose(coord_norm, (0, 2, 1))  
        coord = coord_norm * np.array(self.img_size)
        img_shape = self.img_size[::-1] + (3,)
        kps = [KeypointsOnImage.from_xy_array(coord_JX, shape=img_shape) for coord_JX in coord]  
        cap = cv2.VideoCapture(str(res[PG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/v_fps)
        for n in range(v_size):
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            pOnImg = kps[n]
            img_kps = pOnImg.draw_on_image(re_img)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", img_kps)
            cv2.waitKey(duration)
        cap.release()

