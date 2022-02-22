import math
import numpy as np
from .ml_base import MLBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


class RF(MLBase):
    def __init__(self, cfg, output_path, 
                 X_train=None, y_train=None, 
                 X_test=None, y_test=None):
        super().__init__(cfg, output_path, X_train, y_train, X_test, y_test)
        rf_cfg = cfg["TRAIN"]["RF"]
        if cfg["TRAIN"]["GRIDSEARCH"]:
            rf_model = RandomForestClassifier()()
            hyperparams = {
                "n_estimators": rf_cfg["N_ES"],
                "max_depth": rf_cfg["MAX_DEPTH"],
                "min_samples_split": rf_cfg["MIN_SPLIT"],
                "min_samples_leaf": rf_cfg["MIN_LEAF"],
                "max_features": rf_cfg["MAX_FEATURES"]
            }
            self.model = GridSearchCV(rf_model, hyperparams,
                                        n_jobs=cfg["WORKERS"],
                                        cv=rf_cfg["CV"],
                                        verbose=2)
        else:
            self.model = RandomForestClassifier(n_estimators=rf_cfg["N_ES"],
                                                max_depth=rf_cfg["MAX_DEPTH"],
                                                min_samples_split=rf_cfg["MIN_SPLIT"],
                                                min_samples_leaf=rf_cfg["MIN_LEAF"],
                                                max_features=rf_cfg["MAX_FEATURES"])
            
    def predict(self):
        pred = self.model.predict(self.X_test)
        return pred, np.full((len(pred),), -1)
    