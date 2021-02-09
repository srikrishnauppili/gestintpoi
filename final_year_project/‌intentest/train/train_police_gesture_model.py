from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from pgdataset.s3_handcraft import PgdHandcraft
from constants.enum_keys import PG
from models.gesture_recognition_model import GestureRecognitionModel
from torch import optim
from constants import settings


class Trainer:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest
        self.batch_size = 1
        self.clip_len = 5*7
        pgd = PgdHandcraft(Path.home() / 'intentlong', True, (512, 512), clip_len=self.clip_len)
        self.data_loader = DataLoader(pgd, batch_size=self.batch_size, shuffle=False, num_workers=settings.num_workers)
        self.model = GestureRecognitionModel(batch=self.batch_size)
        self.model.train()
        self.loss = CrossEntropyLoss() 
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 0
        self.model.load_ckpt()
        for epoch in range(100000):
            for ges_data in self.data_loader:  
                features = torch.cat((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                                      ges_data[PG.BONE_ANGLE_SIN]), dim=2)
                features = features.permute(1, 0, 2)  
                features = features.to(self.model.device, dtype=torch.float32)
                h0, c0 = self.model.h0(), self.model.c0()

                _, h, c, class_out = self.model(features, h0, c0)
                target = ges_data[PG.GESTURE_LABEL]
                target = target.to(self.model.device, dtype=torch.long)
                target = target.permute(1, 0)
                target = target.reshape((-1))  
                loss_tensor = self.loss(class_out, target)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                if step % 5000 == 0 and step != 0:
                    self.model.save_ckpt()
                if self.is_unittest:
                    break
                step = step + 1
            if self.is_unittest:
                break
#p1 = Trainer()
#p1.train()
