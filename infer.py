import os
import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from lib.net import NETS
from lib.tools import infer


def main(cfg, opt):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]
    
    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start predicting using device: %s" % device)
    print("Config:", cfg)
        
    if cfg["MODEL"]["NAME"] in NETS:
        Net = NETS[cfg["MODEL"]["NAME"]]
        model = Net(channels=cfg["DATASET"]["CHANNELS"], 
                    joints=cfg["DATASET"]["JOINTS"], 
                    num_cls=cfg["DATASET"]["NUM_CLASSES"])
        model.to(device)
        weights = torch.load(cfg["TEST"]["WEIGHTS"], map_location=device)
        model.load_state_dict(weights)
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["MODEL"]["NAME"])
        
    f = open(opt.file, "r", encoding="utf8")
    lines = f.read().splitlines()[:cfg["DATASET"]["JOINTS"]]
    data = list()
    for line in lines:
        point = line.strip().split()
        assert len(point) == cfg["DATASET"]["CHANNELS"], \
            "Example file is incorrect!"
        point = list(map(float, point))
        data.append(point)
        
    data = np.array(data).astype(np.float32)
    data[:, 0] -= np.min(data[:, 0])
    data[:, 1] -= np.min(data[:, 1])
    data = torch.Tensor(data[None])

    output_label, output, latency = infer(model, data, device, opt.test_speed)
        
    print("Done predicting!")
    print("Label:", output_label)
    print("Label prob:", output)
    print("Latency:", latency)
    print("FPS:", 1.0/latency)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='configs/exam_ds/fc_net.yaml', 
                        help='path to config file')
    parser.add_argument('--file', 
                        type=str, 
                        default='test_hand_reach_out.txt', 
                        help='path to test file')
    parser.add_argument('--test-speed', 
                        type=int, 
                        default=100, 
                        help='run n times to test speed')
    opt = parser.parse_args()
    
    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
            
    main(cfg, opt)
    