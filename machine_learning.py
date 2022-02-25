import os
import argparse
import yaml
import datetime
import json
import pickle
import numpy as np
from time import time
from lib.machine_learning import MODELS
from lib.datasets import DATASETS


def main(cfg, opt, output_path):
    model = MODELS[cfg["MODEL"]["NAME"].lower()]
    ds = DATASETS[cfg["DATASET"]["NAME"]]
    train_cache_file = os.path.join(cfg["DATASET"]["TRAIN"], "data.pkl")
    if cfg["DATASET"]["CACHE"] and os.path.exists(train_cache_file):
            X_train, y_train = pickle.load(open(train_cache_file, "rb"))
            print("Loaded cached data from", train_cache_file)
    else:
        train_ds_loader = ds(data_path=cfg["DATASET"]["TRAIN"],
                        is_train=True,
                        channels=cfg["DATASET"]["CHANNELS"],
                        joints=cfg["DATASET"]["JOINTS"])
        X_train = []
        y_train = []
        for pose, label in train_ds_loader:
            X_train.append(list(pose.flatten()))
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        pickle.dump((X_train, y_train), open(train_cache_file, "wb"))
        print("Saved cached data to", train_cache_file)
        
    val_cache_file = os.path.join(cfg["DATASET"]["VAL"], "data.pkl")
    if cfg["DATASET"]["CACHE"] and os.path.exists(val_cache_file):
            X_test, y_test = pickle.load(open(val_cache_file, "rb"))
            print("Loaded cached data from", val_cache_file)
    else:  
        val_ds_loader = ds(data_path=cfg["DATASET"]["VAL"],
                    is_train=False,
                    channels=cfg["DATASET"]["CHANNELS"],
                    joints=cfg["DATASET"]["JOINTS"])
        X_test = []
        y_test = []
        for pose, label in val_ds_loader:
            X_test.append(list(pose.flatten()))
            y_test.append(label)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        pickle.dump((X_test, y_test), open(val_cache_file, "wb"))
        print("Saved cached data to", val_cache_file)
    
    print("Data shape:", X_train.shape, y_train.shape, 
          X_test.shape, y_test.shape)
        
    if opt.mode.lower() == "train":
        engine = model(cfg, output_path,
                       X_train=X_train, y_train=y_train,
                       X_test=X_test, y_test=y_test)
        val_acc, val_clf_report, train_acc, train_clf_report = engine.train()
        with open(os.path.join(output_path, "train.txt"), "w") as file:
            file.write("Val accuracy: %f\n\n" % val_acc)
            file.write(val_clf_report)
            file.write("\n")
            file.write("Train accuracy: %f\n\n" % train_acc)
            file.write(train_clf_report)
            file.write("\n")
            file.close()
        
    elif opt.mode.lower() == "test":
        engine = model(cfg, output_path,
                       X_test=X_test, y_test=y_test)
        engine.load_model()
        acc, clf_report = engine.test()
        with open(os.path.join(output_path, "test.txt"), "w") as file:
            file.write("Accuracy: %f\n\n" % acc)
            file.write(clf_report)
            file.write("\n")
            file.close()
            
    elif opt.mode.lower() == "predict":
        f = open(opt.file, "r", encoding="utf8")
        lines = f.read().splitlines()[:cfg["DATASET"]["JOINTS"]]
        data = list()
        for line in lines:
            point = line.strip().split()
            assert len(point) == cfg["DATASET"]["CHANNELS"], \
                "Example file is incorrect!"
            point = list(map(float, point))
            data.append(point)
        engine = model(cfg, output_path, X_test=data)
        engine.load_model()
        begin = time()
        res = engine.predict(np.array(data).flatten())
        sps = len(lines)/(time()-begin) # sentences per second
        print(res)
        print("Poses per second:", sps)
    else:
        raise NotImplementedError("%s mode is not implemented!" % opt.mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='predict', 
                        help='train, test or predict?')
    parser.add_argument('--cfg', type=str, default='configs/exam_ds/svm.yaml', 
                        help='path to config file')
    parser.add_argument('--file', type=str, default='test.txt', 
                        help='path to data file for predict')
    opt = parser.parse_args()
    
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
            
    datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
    output_path = os.path.join(os.path.join(cfg["OUTPUT"], opt.mode), 
                               cfg["MODEL"]["NAME"] + "--" +
                               cfg["DATASET"]["NAME"] + 
                               datetime_str)
    os.makedirs(output_path, exist_ok=False)    
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)
        
    main(cfg, opt, output_path)
