from pathlib import Path

import cv2
import numpy as np
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text
from warnings import warn
from constants.enum_keys import PG
from pgdataset.s1_skeleton import PgdSkeleton
from aichallenger.s1_resize import ResizeKeepRatio
import pred.gesture_pred

class Player:
    def __init__(self, is_unittest=False):
        self.img_size = (512, 512)
        self.gpred = pred.gesture_pred.GesturePred()
        self.is_unittest = is_unittest

    def play_dataset_video(self, is_train, video_index, show=True):
        self.scd = PgdSkeleton(Path.home() / 'intentlong', is_train, self.img_size)
        res = self.scd[video_index]
        print(res[PG.COORD_NATIVE].shape)
        print('Playing %s' % res[PG.VIDEO_NAME])
        coord_norm_FXJ = res[PG.COORD_NORM]  
        coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1)) 
        coord = coord_norm_FJX * np.array(self.img_size)
        img_shape = self.img_size[::-1] + (3,)
        kps = [KeypointsOnImage.from_xy_array(coord_JX, shape=img_shape) for coord_JX in coord]  
        cap = cv2.VideoCapture(str(res[PG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/(v_fps*4))
        gestures = [] 
        for n in range(v_size):
            gdict = self.gpred.from_skeleton(coord_norm_FXJ[n][np.newaxis])
            gesture = gdict[PG.OUT_ARGMAX]
            gestures.append(gesture)
            if not show:
                continue
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            ges_name = self.gesture_dict[gesture]
            re_img = draw_text(re_img, 50, 100, ges_name, (255, 50, 50), size=40)
            pOnImg = kps[n]
            img_kps = pOnImg.draw_on_image(re_img)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", img_kps)
            cv2.waitKey(500)
        cap.release()
        gestures = np.array(gestures, np.int)
        res[PG.PRED_GESTURES] = gestures
        print('The prediction of video ', res[PG.VIDEO_NAME], ' is completed')
        return res

    def play_custom_video(self, video_path):
        rkr = ResizeKeepRatio((512, 512))
        if video_path is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise IOError('Failed to open camera.')
        else:
            cap = cv2.VideoCapture(str(video_path))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if v_fps != 15:
                warn('Suggested video frame rate is 15, currently %d, which may impact accuracy' % v_fps)
        duration = 10
        while True:
            ret, img = cap.read()
            if not ret:
                break
            re_img, _, _ = rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            gdict = self.gpred.from_img(re_img)
            gesture = gdict[PG.OUT_ARGMAX]

            coord_norm_FXJ = gdict[PG.COORD_NORM]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            ges_name = self.gesture_dict[gesture]
            re_img = draw_text(re_img, 50, 100, ges_name, (255, 50, 50), size=40)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", re_img)
            cv2.waitKey(duration)
        cap.release()
    def play_custom_image(self, img_path):
        rkr = ResizeKeepRatio((512, 512))
        img = cv2.imread(img_path)
        re_img, _, _ = rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
        gdict = self.gpred.from_img(re_img)
        gesture = gdict[PG.OUT_ARGMAX]
        coord_norm_FXJ = gdict[PG.COORD_NORM]
        coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  
        coord_FJX = coord_norm_FJX * np.array(self.img_size)
        koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
        re_img = koi.draw_on_image(re_img)
        ges_name = self.gesture_dict[gesture]
        re_img = draw_text(re_img, 50, 100, ges_name, (255, 50, 50), size=40)
        cv2.imshow("Play saved keypoint results", re_img)

    gesture_dict = {
        0: "CROSSING",
        1: "NOT_CROSSING",
      }
