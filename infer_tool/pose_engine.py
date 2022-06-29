import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


class UdpPoseAbs(metaclass=ABCMeta):
    
    SKELETONS = {"coco":[
                    [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], 
                    [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], 
                    [3,5], [4,6], [5,7]
                ],
                "mpii": [
                    [9, 10], [12, 13], [12, 11], [3, 2], [2, 1], [14, 15], 
                    [15, 16], [4, 5], [5, 6], [9, 8], [8, 7], [7, 3], [7, 4], 
                    [9, 13], [9, 14]
                ]}
    
    def __init__(self, input_shape, data_type="coco"):
        self.input_shape = input_shape
        
        try:
            self.skeleton = UdpPoseAbs.SKELETONS[data_type]
        except KeyError:
            self.skeleton = None
        self.data_type = "coco"
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
    @staticmethod    
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals
    
    def draw_keypoints(self, image, keypoints, hands=False, radius=5):
        if len(keypoints) == 0:
            return image
        for kpts in keypoints:
            np_kpts = kpts.copy()
            kpts = [list(map(int, x)) for x in kpts]
            if self.skeleton:
                for kid1, kid2 in self.skeleton:
                    if kid1 >= 13 or kid2 >= 13: continue
                    cv2.line(image, tuple(kpts[kid1-1]), tuple(kpts[kid2-1]), (0, 255, 0), 2, cv2.LINE_AA)
            i = 0
            for x, y in kpts:
                if i >= 13: continue
                cv2.circle(image, (x, y), radius, (255, 0, 0), 5, cv2.LINE_AA)
                # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                #     1, (0, 0, 0))
                i += 1
            if hands and self.data_type == "coco":
                lhand = np_kpts[9] + (np_kpts[9]-np_kpts[7])/3
                rhand = np_kpts[10] + (np_kpts[10]-np_kpts[8])/3
                for point in (lhand, rhand):
                    cv2.circle(image, (int(point[0]), int(point[1])), 
                        radius, (0, 0, 255), 2, cv2.LINE_AA)
        return image
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # normalizes:
        img /= 255.0
        img[:, :, 0] = (img[:, :, 0] - self.mean[0]) / self.std[0]
        img[:, :, 1] = (img[:, :, 1] - self.mean[1]) / self.std[1]
        img[:, :, 2] = (img[:, :, 2] - self.mean[2]) / self.std[2]
        return img.transpose(2, 0, 1)
    
    def _postprocess(self, heatmaps):
        preds, maxvals = UdpPoseAbs.get_max_preds(heatmaps)
        return preds, maxvals
    
    @abstractmethod
    def infer_pose(self, person_crop_image):
        pass

    
class UdpPoseOnnx(UdpPoseAbs):
    
    def __init__(self, model_path, input_shape, data_type="coco"):
        super(UdpPoseOnnx, self).__init__(input_shape, data_type)
        
        import onnxruntime
        
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
    
    def infer_pose(self, img):
        pose_input = self._preprocess(img)
        pose_input = np.array(pose_input)[None]
        output = self.ort_session.run(None, {self.input_name: pose_input})[0]
        keypoints, maxvals = self._postprocess(output)
        return keypoints, maxvals, output.shape


class UdpPoseMnn(UdpPoseAbs):
    
    def __init__(self, model_path, input_shape, data_type="coco", 
        heatmap_shape=(1, 17, 64, 48)):
        super(UdpPoseMnn, self).__init__(input_shape, data_type)
        
        import MNN
        self.MNNlib = MNN

        self.interpreter = self.MNNlib.Interpreter(model_path)
        self.interpreter.setCacheFile('.tempcache')
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session, "images") 
        self.heatmap_shape = heatmap_shape
    
    def infer_pose(self, img):
        pose_input = self._preprocess(img)
        pose_input = np.array(pose_input)[None]
        tmp_input = self.MNNlib.Tensor(pose_input.shape, 
            self.MNNlib.Halide_Type_Float, 
            pose_input, 
            self.MNNlib.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session, "output")
        tmp_output = self.MNNlib.Tensor(self.heatmap_shape, 
            self.MNNlib.Halide_Type_Float, 
            np.ones(self.heatmap_shape).astype(np.float32), 
            self.MNNlib.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(tmp_output)
        output = tmp_output.getNumpyData()
        keypoints, maxvals = self._postprocess(output)
        return keypoints, maxvals, output.shape


if __name__ == "__main__":
    file = "img.jpg"
    input_shape = (192, 256)
    # engine = UdpPoseOnnx("weights/shufflenetv2plus_pixel_shuffle_256x192_small.onnx",
    #                         input_shape, "coco")
    engine = UdpPoseMnn("weights/shufflenetv2plus_pixel_shuffle_256x192_small.mnn",
                            input_shape, "coco", (1, 17, 64, 48))
    img = cv2.imread(file)
    keypoints, maxvals, output_shape = engine.infer_pose(img.copy())
    print(keypoints)
    print(maxvals)
    print(output_shape)
    keypoints[:, :, 0] *= (img.shape[1]/output_shape[3])
    keypoints[:, :, 1] *= (img.shape[0]/output_shape[2])
    img = engine.draw_keypoints(img, keypoints, hands=True)
    cv2.imshow("Test", img)
    cv2.waitKey()
    