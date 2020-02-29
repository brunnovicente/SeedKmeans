import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class SKmeans:
    
    def __init__(self, n_grupos = 3):
        self.n_grupos = n_grupos
        
    def fit(self, L, U, y):
        sementes = self.gerarSementes(L, y)
        self.kmeans = KMeans(n_clusters = self.n_grupos, init = sementes, n_init= 1)
        self.kmeans.fit(U)
        
    def predict(self, X):
        return self.kmeans.predict(X)
        
    def gerarSementes(self, X, y):
        data = pd.DataFrame(X)
        data['classe'] = y
        grouped = data.groupby(['classe']).mean()
        return grouped.get_values()    
        