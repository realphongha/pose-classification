import os
import random
import torch
import numpy as np
from tqdm import tqdm
from ..utils.points import angle_between, rotate_kps


class ExamDs2(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, cfg=None, channels=3, joints=13):
        self.data_path = data_path
        self.pose_lbl_path = os.path.join(data_path, "pose_label")
        self.lbl_path = os.path.join(data_path, "label")
        self.labels_file = os.path.join(data_path, "labels.txt")
        self.channels = channels
        self.joints = joints
        self.is_train = is_train
        self.cfg = cfg
        self.data = list()
        self.labels = list()
        self.labels_i = dict()

        self.sample_name = os.listdir(self.pose_lbl_path)

        with open(self.labels_file, "r") as labels_file:
            lbls = labels_file.read().splitlines()
            for lbl in lbls:
                lbl = lbl.strip().split()
                if len(lbl) != 2:
                    continue
                self.labels_i[lbl[0]] = int(lbl[1])

        print("Building dataset...")
        for name in tqdm(self.sample_name):
            label_fn = os.path.join(self.lbl_path, name + ".txt")
            with open(label_fn, "r") as label_file:
                lbls = label_file.read().splitlines()
                for lbl in lbls:
                    lbl = lbl.strip().split()
                    if len(lbl) != 3:
                        continue
                    if lbl[0] not in self.labels_i:
                        print("Wrong label in %s: %s!" % (label_fn, lbl[0]))
                        continue
                    lbl_i, start, end = self.labels_i[lbl[0]], int(lbl[1]), int(lbl[2])
                    if end < start:
                        continue
                    for i in range(start, end+1):
                        fp = os.path.join(os.path.join(self.pose_lbl_path, name), "%i.txt" % i)
                        if os.path.exists(fp):
                            self.data.append(fp)
                            self.labels.append(lbl_i)

        print("Read %i samples from %s" % (len(self.labels), data_path))
        print("Labels:", self.labels_i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pose_file_path, label = self.data[index], self.labels[index]
        with open(pose_file_path, "r") as pose_file:
            pose = pose_file.read().splitlines()[:self.joints]
            pose = [list(map(float, line.strip().split())) for line in pose]
            pose = np.float32(pose)
            pose = pose[:, :self.channels]
        if self.is_train:
            # flips:
            if self.cfg["TRAIN"]["FLIP"] and random.random() < self.cfg["TRAIN"]["FLIP"]:
                pose[:, 0] = 1 - pose[:, 0]
            # random scores:
            if self.cfg["TRAIN"]["RANDOM_SCORE"] and self.channels == 3:
                for point in pose:
                    if random.random() < self.cfg["TRAIN"]["RANDOM_SCORE"]:
                        point[2] = random.random()
        # print(pose)
        # normalizes:
        try:
            if self.cfg["DATASET"]["SPINE_NORMALIZATION"]:
                spine_vector = ((pose[6, :2]+pose[5, :2]-pose[8, :2]-pose[7, :2])/2)
                rotate_rad = angle_between(spine_vector, (0, -1))
                pose[:, :2] = rotate_kps(pose[:, :2], alpha=rotate_rad)
        except KeyError:
            pass
        min0, max0 = np.min(pose[:, 0]), np.max(pose[:, 0])
        min1, max1 = np.min(pose[:, 1]), np.max(pose[:, 1])
        pose[:, 0] = (pose[:, 0]-min0)/(max0-min0)
        pose[:, 1] = (pose[:, 1]-min1)/(max1-min1)
        return pose, label


if __name__ == "__main__":
    ds = ExamDs2(data_path=r'datasets/exam_pose_classification_ds2/val',
                is_train=False,
                channels=3,
                joints=13, 
                cfg={
                    "TRAIN": {
                        "SPINE_NORMALIZATION": False
                    }
                })
    end = 5
    for i in range(end):
        pose, label = ds[i]
        print(pose.shape, pose, label)
        print(list(pose.flatten()))

