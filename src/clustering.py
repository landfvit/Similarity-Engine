import numpy as np
from sklearn.cluster import KMeans

def fit_clusters(data: np.ndarray, k: int = 3) -> KMeans:
    '''
    Train KMeans clustering model on player stats.
    Arguments | data: 2D NumPy array containing standardized player stats.
                      Row -> player, column -> featured stat
                k: number of clusters
    '''
    #init kmeans model
    model = KMeans(
        n_clusters=k, 
        random_state=42, 
        n_init=10 
    )
    model.fit(data)
    
    return model