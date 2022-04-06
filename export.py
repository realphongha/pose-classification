import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from lib.net import NETS


def export_onnx(model, dummy_input, opt):
    import onnx

    torch.onnx.export(model, dummy_input, opt.file, 
                      verbose=False, 
                      opset_version=opt.opset,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Checks
    model_onnx = onnx.load(opt.file)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(opt.file)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(model(dummy_input)), ort_outs[0], rtol=1e-03, atol=1e-05)


def main(opt, cfg):
    if not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(opt.device)
    
    if cfg["MODEL"]["NAME"] in NETS:
        Net = NETS[cfg["MODEL"]["NAME"]]
        model = Net(channels=cfg["DATASET"]["CHANNELS"], 
                    joints=cfg["DATASET"]["JOINTS"], 
                    num_cls=cfg["DATASET"]["NUM_CLASSES"])
        model.to(device)
        weights = torch.load(opt.weights, map_location=device)
        model.load_state_dict(weights)
        model.eval()
        
        dummy_input = torch.zeros(opt.batch, cfg["DATASET"]["JOINTS"], 
                                  cfg["DATASET"]["CHANNELS"]).to(device)
        if opt.format == "onnx":
            export_onnx(model, dummy_input, opt)
        else:
            raise Exception("%s format is not supported!" % opt.format)
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["MODEL"]["NAME"])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--format', type=str, default="onnx", help='format to export')
    parser.add_argument('--file', type=str, required=True, help='filename to export')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    opt = parser.parse_args()
    
    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
            
    main(opt, cfg)