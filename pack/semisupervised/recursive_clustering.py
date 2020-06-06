from ..base import SemiSupervisedBase
from sklearn.metrics.pairwise import euclidean_distances
from ..utils import random_per_target
import numpy as np
import warnings
import datetime
import sys
warnings.filterwarnings('ignore')

class RecursiveClustering(SemiSupervisedBase):
    """
    Example
    >> rc = RecursiveClustering(fp={'target':[], 'centroid':[], 'n':[], 'dt':[]}, th=0., init='random', verbose=True, max_recursive=2000)
    >> rc.fit(x=x, y=y)
    >>

    References
    -
    """

    def __init__(self, fp, th=0., init='random', verbose=True, max_recursive=2000):
        """
        Init class
        :param fp: final partisions
        :param th: threshold
        :param init: initial centroid ('random' or callback)
        :param verbose: show the process
        :param max_recursive: max recursive
        """
        self._th = th
        self._fp = fp
        if init == 'random':
            self._init = random_per_target
        else:
            self._init = init
        self._fp['dt'].append(datetime.datetime.now())
        self._verbose = verbose
        self._max_recursive = max_recursive
        sys.setrecursionlimit(self._max_recursive)

    def fit(self, x, y):
        """
        Learn data
        :param x:
        :param y:
        :param validation_data:
        :return: self
        """

        # convert x y to ndarray
        x = np.array(x)
        y = np.array(y)

        # validate the data
        x, y = self._validate(x, y)

        # find unique target, null or None will be reputed as unlabel data
        y_unique = np.unique(y)
        y_unique = y_unique[y_unique != None]

        # make partitions
        partitions = [[] for c in range(y_unique.shape[0])]

        # clustering proccess
        labels = self._cluster(n_clusters=y_unique.shape[0], x=x, y=y, init_function=self._init)

        # agglomerate data to each suit partition
        for idx, label in enumerate(labels):
            partitions[label].append([idx, y[idx]])

        # convert each partition to ndarray
        for c in range(y_unique.shape[0]):
            partitions[c] = np.array(partitions[c])

        # check every partition
        for partition in partitions:
            # find unique target and n data per target
            target = np.unique(partition[:, 1], return_counts=True)
            n_per_target = target[1]
            target = target[0]

            # find null index and delete them
            unlabel_idx = np.where(target == None)[0]
            target = np.delete(target, unlabel_idx)
            n_per_target = np.delete(n_per_target, unlabel_idx)

            # find max n data index
            highest_target_idx = np.argmax(n_per_target)

            # count relative presentage
            rps = []
            for c in range(target.shape[0]):
                if c != highest_target_idx:
                    rps.append(n_per_target[c] / n_per_target[highest_target_idx])

            # get highest relative presentage
            try:
                highest_rps = np.max(rps)
            except:
                highest_rps = '-'

            if self._verbose:
                v = f'recursives : {len(self._fp["dt"])}x | '
                v += f'partisions : {len(self._fp["target"])}'
                sys.stdout.write(f'\r{v}')

            # do recursion if relative presetage > threshold
            if target.shape[0] > 1 and highest_rps > self._th:
                new_x = x[partition[:, 0].astype(np.int)]
                new_y = partition[:, 1]
                self._recursiveClass(new_x=new_x, new_y=new_y, fp=self._fp, th=self._th, init=self._init, verbose=self._verbose, max_recursive=self._max_recursive)
            else:
                target = target[highest_target_idx]
                centroid = list(x[partition[:, 0].astype(np.int)].mean(axis=0))
                self._fp['target'].append(target)
                self._fp['centroid'].append(centroid)
                self._fp['n'].append(n_per_target[highest_target_idx])

        return self

    def _recursiveClass(self, new_x, new_y, fp, th, init, verbose, max_recursive):
        RecursiveClustering(th=th, fp=fp, verbose=verbose, init=init, max_recursive=max_recursive).fit(new_x, new_y)

    def _cluster(self, n_clusters, x, y, init_function):
        pass

    def _validate(self, x, y):
        unique_x, indices, n_x = np.unique(x, axis=0, return_counts=True, return_index=True)
        return x[indices], y[indices]

    def getFP(self):
        return {'target': np.array(self._fp['target']), 'centroid': np.array(self._fp['centroid']),
                'n': np.array(self._fp['n']), 'dt': np.array(self._fp['dt'])}

    def predict(self, x):
        x = np.array(x)
        fp = self.getFP()
        distances = euclidean_distances(x, fp['centroid'])
        y = []
        for d in distances:
            y.append(fp['target'][np.argmin(d)])
        return np.array(y)