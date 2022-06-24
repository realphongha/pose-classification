import os
import sys
import argparse
import requests
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from lib.net import NETS
from lib.tools import infer
from infer_tool.objdet_engine import *
from infer_tool.pose_engine import *


def main(cfg, opt):
    ACTION_NAME = ["Hand reach out", "Look down", "Look outside", "Sitting"]
    if opt.objdet.endswith(".mnn"):
        objdet_engine = NanoDetMnn(opt.objdet, opt.device, (320, 320))
    else:
        raise NotImplementedError

    if opt.pose.endswith(".mnn"):
        pose_engine = UdpPsaPoseMnn(opt.pose, (192, 256))

    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]

    if cfg["MODEL"]["NAME"] in NETS:
        Net = NETS[cfg["MODEL"]["NAME"]]
        model = Net(channels=cfg["DATASET"]["CHANNELS"],
                    joints=cfg["DATASET"]["JOINTS"],
                    num_cls=cfg["DATASET"]["NUM_CLASSES"])
        model.to(opt.device)
        weights = torch.load(cfg["TEST"]["WEIGHTS"], map_location=opt.device)
        model.load_state_dict(weights)
    else:
        raise NotImplementedError("%s is not implemented!" %
                                  cfg["MODEL"]["NAME"])

    if opt.source == "0":
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            h, w = frame.shape[:2]
            bboxes = objdet_engine.infer(frame)
            if not bboxes: continue
            best_bbox = None
            best_score = -1
            for bbox in bboxes:
                if int(bbox[5]) != 0: continue
                if bbox[4] > best_score:
                    best_score = bbox[4]
                    best_bbox = bbox
            if best_bbox is None:
                continue
            x1, y1, x2, y2 = best_bbox[:4]
            x1 = round(x1/objdet_engine.input_shape[0]*w)
            x2 = round(x2/objdet_engine.input_shape[0]*w)
            y1 = round(y1/objdet_engine.input_shape[1]*h)
            y2 = round(y2/objdet_engine.input_shape[1]*h)
            pose_img = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            pose, _, output_shape = pose_engine.infer_pose(pose_img)
            pose = pose[0]
            real_pose = pose.copy()
            real_pose[:, 0] *= pose_img.shape[1] / output_shape[3]
            real_pose[:, 1] *= pose_img.shape[0] / output_shape[2]
            real_pose[:, 0] += x1
            real_pose[:, 1] += y1
            pose_engine.draw_keypoints(frame, real_pose.astype(int)[None])
            data = pose[:13].astype(np.float32)
            min0, max0 = np.min(data[:, 0]), np.max(data[:, 0])
            min1, max1 = np.min(data[:, 1]), np.max(data[:, 1])
            data[:, 0] = (data[:, 0]-min0)/(max0-min0)
            data[:, 1] = (data[:, 1]-min1)/(max1-min1)
            data = torch.Tensor(data[None])
            output_label, output, raw_output, latency = infer(model, data, opt.device)
            print("Label:", ACTION_NAME[output_label[0]])
            print("Label prob:", output[0])
            cv2.putText(frame, 
                "%s %.2f" % (ACTION_NAME[output_label[0]], output[0][output_label[0]]), 
                (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif opt.source == "1":
        skip = 10
        i = 0
        while True:
            i += 1
            if i % skip != 0: continue
            img_resp = requests.get(opt.ip)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            h, w = frame.shape[:2]
            bboxes = objdet_engine.infer(frame)
            if not bboxes: continue
            best_bbox = None
            best_score = -1
            for bbox in bboxes:
                if int(bbox[5]) != 0: continue
                if bbox[4] > best_score:
                    best_score = bbox[4]
                    best_bbox = bbox
            if best_bbox is None:
                continue
            x1, y1, x2, y2 = best_bbox[:4]
            x1 = round(x1/objdet_engine.input_shape[0]*w)
            x2 = round(x2/objdet_engine.input_shape[0]*w)
            y1 = round(y1/objdet_engine.input_shape[1]*h)
            y2 = round(y2/objdet_engine.input_shape[1]*h)
            x1 -= 5
            y1 -= 5
            x2 += 5
            y2 += 5
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = w-1 if x2 >= w else x2
            y2 = h-1 if y2 >= h else y2
            pose_img = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            pose, _, output_shape = pose_engine.infer_pose(pose_img)
            pose = pose[0]
            real_pose = pose.copy()
            real_pose[:, 0] *= pose_img.shape[1] / output_shape[3]
            real_pose[:, 1] *= pose_img.shape[0] / output_shape[2]
            real_pose[:, 0] += x1
            real_pose[:, 1] += y1
            pose_engine.draw_keypoints(frame, real_pose.astype(int)[None])
            data = pose[:13].astype(np.float32)
            min0, max0 = np.min(data[:, 0]), np.max(data[:, 0])
            min1, max1 = np.min(data[:, 1]), np.max(data[:, 1])
            data[:, 0] = (data[:, 0]-min0)/(max0-min0)
            data[:, 1] = (data[:, 1]-min1)/(max1-min1)
            data = torch.Tensor(data[None])
            output_label, output, raw_output, latency = infer(model, data, opt.device)
            print("Label:", ACTION_NAME[output_label[0]])
            print("Label prob:", output[0])
            frame = cv2.resize(frame, (640, 360))
            cv2.putText(frame, 
                "%s %.2f" % (ACTION_NAME[output_label[0]], output[0][output_label[0]]), 
                (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/exam_ds2/fc_net_2_channels.yaml',
                        help='path to config file')
    parser.add_argument('--source',
                        type=str,
                        default='1',
                        help='0 for webcam, 1 for IP server')
    parser.add_argument('--ip',
                        type=str,
                        default='http://192.168.0.135:8080/shot.jpg',
                        help='IP server address')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device for inference')
    parser.add_argument('--objdet',
                        type=str,
                        default='weights/nanodet_plus_m_320.mnn',
                        help='object detection weights path')
    parser.add_argument('--pose',
                        type=str,
                        default='weights/shufflenetv2plus_pixel_shuffle_256x192_small.mnn',
                        help='pose estimation weights path')
    opt = parser.parse_args()

    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    main(cfg, opt)
