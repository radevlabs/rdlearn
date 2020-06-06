from sklearn.metrics import accuracy_score
import numpy as np

class SemiSupervisedBase:
    def fit(self, x, y):
        return self

    def predict(self, x):
        pass

    def score(self, x, y):
        return accuracy_score(y, self.predict(x))