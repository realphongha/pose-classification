import os
import datetime
import json

import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from lib.datasets import DATASETS
from lib.net import NETS
from lib.tools import evaluate


def main(cfg, output_path):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]

    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start testing using device: %s" % device)
    print("Config:", cfg)

    if cfg["DATASET"]["NAME"] in DATASETS:
        Ds = DATASETS[cfg["DATASET"]["NAME"]]
        val_ds = Ds(data_path=cfg["DATASET"]["VAL"],
                    is_train=False,
                    cfg=cfg,
                    channels=cfg["DATASET"]["CHANNELS"],
                    joints=cfg["DATASET"]["JOINTS"])
    else:
        raise NotImplementedError("%s is not implemented!" %
                                  cfg["DATASET"]["NAME"])

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

    val_loader = DataLoader(val_ds,
                            batch_size=cfg["TEST"]["BATCH_SIZE"],
                            shuffle=cfg["TEST"]["SHUFFLE"],
                            num_workers=cfg["WORKERS"])

    criterion = CrossEntropyLoss()

    f1, acc, clf_report, loss = evaluate(model, criterion,
                                     val_loader, device, log=False)
    print("Done testing!")
    print("Macro avg F1 score:", f1)
    print(clf_report)
    with open(os.path.join(output_path, "final_results.txt"), "w") as file:
        file.write("Macro avg F1 score: %f\n\n" % f1)
        file.write(clf_report)
        file.write("\n")
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/exam_ds/fc_net.yaml',
                        help='path to config file')
    opt = parser.parse_args()

    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
    output_path = os.path.join(os.path.join(cfg["OUTPUT"], "test"),
                               cfg["MODEL"]["NAME"] + "--" +
                               cfg["DATASET"]["NAME"] +
                               datetime_str)
    os.makedirs(output_path, exist_ok=False)
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)

    main(cfg, output_path)
