import os
import glob
import json
import numpy as np
from sklearn.cluster import KMeans

from cluster_models.clustering_model import ClusteringModel


class KMeansClusteringModel(ClusteringModel):
    def __init__(self, encoded_vectors_dir, random_seed=None):
        self.encoded_vectors_dir = encoded_vectors_dir
        self.encoded_vectors = []
        self.encoded_vectors_map = {}
        self.kmeans = None
        self.random_seed = random_seed
        
    def load_encoded_vectors_files(self, filenames):
        for filename in filenames:
            basename = filename.split('/')[-1].split('.')[0]
            
            with open(filename, 'r') as fh:
                vector = json.load(fh)
                self.encoded_vectors.append(vector)
                self.encoded_vectors_map[tuple(vector)] = basename
                
        self.encoded_vectors = np.array(self.encoded_vectors)
        
        return self.encoded_vectors, self.encoded_vectors_map
        
    def cluster(self, x, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed).fit(x)
        return self.kmeans

    def get_stockname_label_pairs(self, stocknames):
        if self.kmeans is None:
            raise Exception("must run cluster() first")

        clean_stocknames = [stockname.split('/')[-1].split('.')[0] for stockname in stocknames]
        return sorted(zip(clean_stocknames, self.kmeans.labels_), key=lambda x: x[1]) 

