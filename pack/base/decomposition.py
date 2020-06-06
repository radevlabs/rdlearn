from sklearn.decomposition.pca import PCA
class DecompositionBase:
    def __init__(self, n_features=2):
        self._n_features = n_features

    def fit(self, x):
        pass

    def transform(self, x):
        pass