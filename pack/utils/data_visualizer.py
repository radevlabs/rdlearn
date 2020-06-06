from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(x, y, title='Data visualization'):
    x = np.array(x)
    y = np.array(y)
    colors = ['blue', 'green', 'red', 'brown', 'yellow', 'black', 'cyan', 'magenta']
    if x.shape[1] != 2:
        x = TSNE(n_components=2).fit_transform(x)
    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
    plt.figure(figsize=(10, 10))
    for c, label in enumerate(np.unique(y)):
        index = np.where(y == label)[0]
        plt.scatter(x=x[index, 0],
                    y=x[index, 1],
                    c=colors[c],
                    label=label,
                    marker='o')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(title)
    return plt.show(), x