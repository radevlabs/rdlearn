from .recursive_clustering import RecursiveClustering
from sklearn.cluster import KMeans

class RKMeans(RecursiveClustering):
    def _cluster(self, n_clusters, x, y, init_function):
        kmeans = KMeans(n_clusters=n_clusters, init=init_function(x=x, y=y))
        kmeans.fit(x)
        return kmeans.predict(x)

    def _recursiveClass(self, new_x, new_y, fp, th, init, verbose, max_recursive):
        RKMeans(th=th, fp=fp, verbose=verbose, init=init, max_recursive=max_recursive).fit(new_x, new_y)

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x = load_iris()
    y = x['target_names']
    y = y[x['target']]
    x = x['data']

    rkmeans = RKMeans(fp={'target': [], 'centroid': [], 'n': [], 'dt': []}, th=0., init='random', verbose=True,
                      max_recursive=2000)
    rkmeans.fit(x, y)
    print('')
    print(rkmeans.score(x, y))