import math
import numpy as np
from .ml_base import MLBase
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


class SVM(MLBase):
    def __init__(self, cfg, output_path, 
                 X_train=None, y_train=None, 
                 X_test=None, y_test=None):
        super().__init__(cfg, output_path, X_train, y_train, X_test, y_test)
        svm_cfg = cfg["TRAIN"]["SVM"]
        if cfg["TRAIN"]["GRIDSEARCH"]:
            svm_model = SVC()
            hyperparams = {
                "C": svm_cfg["C"],
                "kernel": svm_cfg["KERNEL"],
                "gamma": svm_cfg["GAMMA"]
            }
            self.model = GridSearchCV(svm_model, hyperparams,
                                        n_jobs=cfg["WORKERS"],
                                        cv=svm_cfg["CV"],
                                        verbose=2)
        else:
            self.model = SVC(C=svm_cfg["C"], kernel=svm_cfg["KERNEL"],
                                gamma=svm_cfg["GAMMA"])
            
    def predict(self):
        pred = self.model.predict(self.X_test)
        y_margins = self.model.decision_function(self.X_test)
        return pred, np.amax(sigmoid_v(y_margins), axis=1)
    