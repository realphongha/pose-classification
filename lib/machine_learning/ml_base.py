import pickle
import os
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class MLBase(ABC):
    def __init__(self, cfg, output_path,
                 X_train=None, y_train=None, 
                 X_test=None, y_test=None):
        self.cfg = cfg
        self.output_path = output_path
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        if self.cfg["TRAIN"]["GRIDSEARCH"]:
            best_params_str = "Best params: " + str(self.model.best_params_)
            print(best_params_str)
            with open(os.path.join(self.output_path, "best_params.txt"), "w") as f:
                f.write(best_params_str)
            self.model = self.model.best_estimator_
        self.save_model()
        pred_train = self.model.predict(self.X_train)
        pred_val = self.model.predict(self.X_test)
        print("Result on valset:")
        val_acc = accuracy_score(self.y_test, pred_val)
        print(val_acc)
        matrix = confusion_matrix(self.y_test, pred_val)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        val_clf_report = classification_report(self.y_test, pred_val)
        print(val_clf_report)
        print("Result on trainset:")
        train_acc = accuracy_score(self.y_train, pred_train)
        print(train_acc)
        matrix = confusion_matrix(self.y_train, pred_train)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        train_clf_report = classification_report(self.y_train, pred_train)
        print(train_clf_report)
        return val_acc, val_clf_report, train_acc, train_clf_report
    
    def test(self):
        pred = self.model.predict(self.X_test)
        print("Result on valset:")
        acc = accuracy_score(self.y_test, pred)
        print(acc)
        matrix = confusion_matrix(self.y_test, pred)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(matrix)
        print(accs) 
        clf_report = classification_report(self.y_test, pred)
        print(clf_report)
        return acc, clf_report

    @abstractmethod
    def predict(self):
        pass

    def save_model(self):
        pickle.dump(self.model, open(os.path.join(self.output_path, "model.pkl"), "wb"))
        print("Saved model to %s" % os.path.join(self.output_path, "model.pkl"))
        
    def load_model(self):
        f = open(self.cfg["TEST"]["WEIGHTS"], 'rb')
        self.model = pickle.load(f)
        print("Loaded model from %s" % self.cfg["TEST"]["WEIGHTS"])
