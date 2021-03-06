import os
import datetime
import json

import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if cfg["GPUS"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]

    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start training using device: %s" % device)
    print("Config:", cfg)

    if cfg["DATASET"]["NAME"] in DATASETS:
        Ds = DATASETS[cfg["DATASET"]["NAME"]]
        train_ds = Ds(data_path=cfg["DATASET"]["TRAIN"],
                      is_train=True,
                      cfg=cfg,
                      channels=cfg["DATASET"]["CHANNELS"],
                      joints=cfg["DATASET"]["JOINTS"])
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

    best_f1 = -1
    best_clf_report = None
    best_conf_matrix = None
    best_ckpt = os.path.join(output_path, "best.pth")
    decreasing = 0

    train_loss = list()
    val_loss = list()
    train_f1 = list()
    val_f1 = list()

    for epoch in range(cfg["TRAIN"]["EPOCHS"]):
        print("EPOCH %i:" % epoch)
        f1, acc, loss, conf_matrix = train(model, criterion, optimizer, train_loader, device)
        train_f1.append(f1)
        train_loss.append(loss)
        lr_scheduler.step()
        f1, acc, clf_report, loss, conf_matrix = evaluate(model, criterion,
                                                          val_loader, device)
        val_f1.append(f1)
        val_loss.append(loss)
        if f1 > best_f1:
            best_f1 = f1
            best_clf_report = clf_report
            best_conf_matrix = conf_matrix
            torch.save(model.state_dict(), best_ckpt)
            print("Saved checkpoint to", best_ckpt)
            decreasing = 0
        else:
            decreasing += 1
            if cfg["TRAIN"]["EARLY_STOPPING"] and decreasing > cfg["TRAIN"]["EARLY_STOPPING"]:
                print("Early stopped!")
                break
        print()

    print("Done training!")
    print("Best macro avg f1 score:", best_f1)
    print(best_clf_report)
    with open(os.path.join(output_path, "final_results.txt"), "w") as file:
        file.write("Macro avg f1 score: %f\n\n" % best_f1)
        file.write(best_clf_report)
        file.write("\n")
        file.close()

    epochs = range(epoch+1)

    fig = plt.figure()
    plt.plot(epochs, train_f1, 'r', label='Training F1')
    plt.plot(epochs, val_f1, 'b', label='Validation F1')
    plt.title('Training and validation F1 score')
    plt.legend()
    fig.savefig(os.path.join(output_path, 'f1_plot.png'),
                bbox_inches='tight')

    fig = plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    fig.savefig(os.path.join(output_path, 'loss_plot.png'),
                bbox_inches='tight')

    fig = plt.figure()
    df_cm = pd.DataFrame(best_conf_matrix, range(best_conf_matrix.shape[0]),
                         range(best_conf_matrix.shape[0]))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    fig.savefig(os.path.join(output_path, 'confusion_matrix.png'),
                bbox_inches='tight')


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
    output_path = os.path.join(os.path.join(cfg["OUTPUT"], "train"),
                               cfg["MODEL"]["NAME"] + "--" +
                               cfg["DATASET"]["NAME"] +
                               datetime_str)
    os.makedirs(output_path, exist_ok=False)
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)

    main(cfg, output_path)
