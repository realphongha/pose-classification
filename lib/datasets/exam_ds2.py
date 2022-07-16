import os
import random
from matplotlib.pyplot import sca
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
            # rotates:
            try:
                min_degree, max_degree = self.cfg["TRAIN"]["ROTATE"]
                degree = random.randint(min_degree, max_degree)
                pose[:, :2] = rotate_kps(pose[:, :2], degrees=degree)
            except KeyError:
                pass
            except Exception as e:
                print("Exception when rotating keypoints:", e)
            # extends body parts:
            try:
                if self.cfg["TRAIN"]["EXTEND_BODY_PARTS"]:
                    min_perc, max_perc = self.cfg["TRAIN"]["EXTEND_BODY_PARTS"]
                    # forearms:
                    perc = random.randint(min_perc, max_perc)
                    ratio = perc/100
                    pose[10, :2] += (pose[10, :2] - pose[8, :2]) * ratio
                    pose[9, :2] += (pose[9, :2] - pose[7, :2]) * ratio
                    # upper arms:
                    perc = random.randint(min_perc, max_perc)
                    ratio = perc/100
                    pose[10, :2] += (pose[8, :2] - pose[6, :2]) * ratio
                    pose[8, :2] += (pose[8, :2] - pose[6, :2]) * ratio
                    pose[9, :2] += (pose[7, :2] - pose[5, :2]) * ratio
                    pose[7, :2] += (pose[7, :2] - pose[5, :2]) * ratio
                    # body:
                    perc = random.randint(min_perc, max_perc)
                    ratio = perc/100
                    pose[12, :2] += (pose[12, :2] - pose[6, :2]) * ratio
                    pose[11, :2] += (pose[11, :2] - pose[5, :2]) * ratio

            except KeyError:
                pass
            except Exception as e:
                print("Exception when extending body parts:", e)
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
    import cv2
    skeletons = [[16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], 
                 [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], 
                 [3,5], [4,6], [5,7]]
    ds = ExamDs2(data_path=r'datasets/exam_pose_classification_ds2/val',
                is_train=False,
                channels=3,
                joints=13, 
                cfg={
                    "TRAIN": {
                        "SPINE_NORMALIZATION": False,
                        "ROTATE": (-30, 30),
                        "EXTEND_BODY_PARTS": (-20, 20)
                    }
                })
    num = 20
    img_size = 320
    while True:
        num -= 1
        if num < 0: break
        i = random.randrange(0, len(ds))
        pose, label = ds[i]
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        scaled_pose = list()
        for x, y, _ in pose:
            x, y = int(x*img_size), int(y*img_size)
            scaled_pose.append((x, y))
        for kp in scaled_pose:
            cv2.circle(img, kp, 5, (255, 255, 255), 5, cv2.LINE_AA)
        for skeleton in skeletons:
            s, e = skeleton[0]-1, skeleton[1]-1
            if s >= 13 or e >= 13: continue
            cv2.line(img, scaled_pose[s], scaled_pose[e], (0, 255, 0), 1, 
                cv2.LINE_AA)
        print("Label:", label)
        cv2.imshow("Keypoints", img)
        cv2.waitKey()
        # print(pose.shape, pose, label)

