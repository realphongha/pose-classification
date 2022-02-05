import os
import datetime
import json

import argparse
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from lib.datasets import DATASETS
from lib.net import NETS
from lib.tools import train, evaluate


def main(cfg, output_path):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]
    
    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start training using device: %s" % device)
    print("Config:", cfg)
    
    if cfg["DATASET"]["NAME"] in DATASETS:
        Ds = DATASETS[cfg["DATASET"]["NAME"]]
        train_ds = Ds(data_path=cfg["DATASET"]["TRAIN"],
                      is_train=True,
                      channels=cfg["DATASET"]["CHANNELS"],
                      joints=cfg["DATASET"]["JOINTS"])
        val_ds = Ds(data_path=cfg["DATASET"]["VAL"],
                    is_train=False,
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
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["MODEL"]["NAME"])
        
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["TRAIN"]["BATCH_SIZE"],
                              shuffle=cfg["TRAIN"]["SHUFFLE"],
                              num_workers=cfg["WORKERS"])
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["TEST"]["BATCH_SIZE"],
                            shuffle=cfg["TEST"]["SHUFFLE"],
                            num_workers=cfg["WORKERS"])
    
    if cfg["TRAIN"]["OPTIMIZER"] == "adam":
        optimizer = optim.Adam(model.parameters(), 
                               lr=cfg["TRAIN"]["LR"])
    elif cfg["TRAIN"]["OPTIMIZER"] == "sgd":
        optimizer = optim.SGD(model.parameters(), 
                              lr=cfg["TRAIN"]["LR"],
                              momentum=cfg["TRAIN"]["MOMENTUM"],
                              weight_decay=cfg["TRAIN"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError("%s is not implemented!" % 
                                  cfg["TRAIN"]["OPTIMIZER"])
        
    criterion = CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, cfg["TRAIN"]["LR_STEP"], cfg["TRAIN"]["LR_FACTOR"],
    )
    
    best_acc = -1
    best_clf_report = None
    best_ckpt = os.path.join(output_path, "best.pth")
    
    for epoch in range(cfg["TRAIN"]["EPOCHS"]):
        print("EPOCH %i:" % epoch)
        train(model, criterion, optimizer, train_loader, device)
        lr_scheduler.step()
        acc, clf_report = evaluate(model, criterion, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_clf_report = clf_report
            torch.save(model.state_dict(), best_ckpt)
            print("Saved checkpoint to", best_ckpt)
        print()
        
    print("Done training!")
    print("Best accuracy:", best_acc)
    print(best_clf_report)
    with open(os.path.join(output_path, "final_results.txt"), "w") as file:
        file.write("Accuracy: %f\n\n" % best_acc)
        file.write(best_clf_report)
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
    output_path = os.path.join(cfg["OUTPUT"], 
                               cfg["MODEL"]["NAME"] + "--" +
                               cfg["DATASET"]["NAME"] + 
                               datetime_str)
    os.makedirs(output_path, exist_ok=False)    
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)
            
    main(cfg, output_path)