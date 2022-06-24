from abc import ABC, abstractmethod
import time
import math

import cv2
import numpy as np


def iou_calc(boxes1, boxes2):
    boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    left_up = np.maximum(boxes1[:2], boxes2[:2])
    right_down = np.minimum(boxes1[2:-2], boxes2[2:-2])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[0] * inter_section[1]

    return 1.0 * inter_area / (boxes1_area+boxes2_area-inter_area)


def multiclass_nms(boxes, iou_threshold=0.75, conf_threshold=None, 
                   max_nms=1000):
    if boxes.shape[0] == 0:
        return list()
    
    if conf_threshold:
        keep_idx = []
        for i, box in enumerate(boxes):
            score, cls = box[4:]
            if conf_threshold[round(cls)] <= score:
                keep_idx.append(i)
        boxes = boxes[keep_idx]

    # sort by confidence
    sorted_i = boxes[:, 4].argsort()[::-1]
    if boxes.shape[0] > max_nms:
        sorted_i = sorted_i[:max_nms]
    boxes = boxes[sorted_i]
    return_box = []
        
    boxes_dict = {}
    for box in boxes:
        if box[5] in boxes_dict:
            boxes_dict[box[5]].append(box)
        else:
            boxes_dict[box[5]] = [box]

    for boxs in boxes_dict.values():
        if len(boxs) == 1:
            return_box.append(boxs[0])
        else:
            while(boxs):
                best_box = boxs.pop(0)
                return_box.append(best_box)
                j = 0
                for i in range(len(boxs)):
                    i -= j
                    if iou_calc(best_box, boxs[i]) > iou_threshold:
                        boxs.pop(i)
                        j += 1
    return return_box


def generate_grid_center_priors(input_h, input_w, strides):
    center_priors = list()
    for stride in strides:
        feat_w = math.ceil(input_w/stride)
        feat_h = math.ceil(input_h/stride)
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append([x, y, stride])
    return center_priors  


class NanoDetAbs(ABC):
    def __init__(self, model_path, device, input_shape, std, mean, num_cls, 
                 reg_max, strides, score_thres, iou_thres):
        self.model_path = model_path
        self.device = device
        self.input_shape = input_shape
        self.std = std
        self.mean = mean
        self.num_cls = num_cls
        self.reg_max = reg_max
        self.strides = strides
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        
    def _preprocess(self, img):
        img = img.astype(np.float32)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # normalizes:
        img[:, :, 0] = (img[:, :, 0] - self.mean[0]) / self.std[0]
        img[:, :, 1] = (img[:, :, 1] - self.mean[1]) / self.std[1]
        img[:, :, 2] = (img[:, :, 2] - self.mean[2]) / self.std[2]
        # transposes:
        img = img.transpose(2, 0, 1)
        return img
    
    def distance2bbox(self, dfl_pred, x, y, stride):
        ct_x = x*stride
        ct_y = y*stride
        dis_pred = np.zeros((4,))
        # removes for loop to increase performance speed:
        length = self.reg_max+1
        for i in range(4):
            dis = 0
            dis_after_sm = np.zeros((length,))
            idx = i*length
            alpha = max(dfl_pred[idx:idx+length])
            for j in range(length):
                dis_after_sm[j] = math.exp(dfl_pred[idx+j]-alpha)
            dis_after_sm /= np.sum(dis_after_sm)
            for j in range(length):
                dis += j * dis_after_sm[j];
            dis *= stride
            dis_pred[i] = dis
        xmin = max(ct_x-dis_pred[0], 0)
        ymin = max(ct_y-dis_pred[1], 0)
        xmax = min(ct_x+dis_pred[2], self.input_shape[1])
        ymax = min(ct_y+dis_pred[3], self.input_shape[0])
        return [xmin, ymin, xmax, ymax]
    
    def _postprocess(self, output):
        results = list()
        cls_preds, dis_preds = output[:, :self.num_cls], output[:, self.num_cls:]
        
        center_priors = generate_grid_center_priors(self.input_shape[0],
                                                    self.input_shape[1],
                                                    self.strides)
        for i, center in enumerate(center_priors):
            x, y, stride = center
            max_cls = np.argmax(cls_preds[i, :])
            score = cls_preds[i, max_cls]
            if score > self.score_thres[max_cls]:
                bbox = self.distance2bbox(dis_preds[i], x, y, stride)
                bbox.append(score)
                bbox.append(max_cls)
                results.append(bbox)
        results = np.array(results)
        results = multiclass_nms(results, self.iou_thres, None)
        return results
    
    @abstractmethod
    def infer(self, img):
        pass
    
    
class NanoDetOnnx(NanoDetAbs):
    def __init__(self, model_path, device, input_shape, 
        std=[57.375, 57.12, 58.395], 
        mean=[103.53, 116.28, 123.675], 
        num_cls=5, 
        reg_max=7, 
        strides=[8, 16, 32, 64], 
        score_thres=[0.5, 0.2, 0.2, 0.2, 0.2], 
        iou_thres=0.2):
        super().__init__(model_path, device, input_shape, std, mean, num_cls, 
                         reg_max, strides, score_thres, iou_thres)
        
        import onnxruntime
        
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
       
    def infer(self, img):
        model_input = self._preprocess(img)
        model_input = model_input[None]
        begin = time.time()
        output = self.ort_session.run(None, {self.input_name: model_input})
        print("Latency: %.2f(ms)" % (time.time()-begin))
        output = output[0][0]
        bboxes = self._postprocess(output)
        return bboxes

class NanoDetMnn(NanoDetAbs):
    import MNN

    def __init__(self, model_path, device, input_shape, 
        std=[57.375, 57.12, 58.395], 
        mean=[103.53, 116.28, 123.675], 
        num_cls=5, 
        reg_max=7, 
        strides=[8, 16, 32, 64], 
        score_thres=[0.5, 0.2, 0.2, 0.2, 0.2], 
        iou_thres=0.2):
        super().__init__(model_path, device, input_shape, std, mean, num_cls, 
                         reg_max, strides, score_thres, iou_thres)
        
        self.interpreter = self.MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
       
    def infer(self, img):
        model_input = self._preprocess(img)
        model_input = model_input[None]
        begin = time.time()
        tmp_input = self.MNN.Tensor(model_input.shape, 
            self.MNN.Halide_Type_Float, 
            model_input, 
            self.MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output = self.interpreter.getSessionOutput(self.session).getNumpyData()
        print("Latency: %.2f(ms)" % (time.time()-begin))
        output = output[0]
        # print(output)
        # print(output.shape)
        bboxes = self._postprocess(output)
        return bboxes
