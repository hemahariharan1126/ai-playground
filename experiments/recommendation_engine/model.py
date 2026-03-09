import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

class RecommendationEngine:
    def __init__(self, n_components=10):
        self.svd = TruncatedSVD(n_components=n_components)
        self.matrix = None
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """
        Fits the SVD model to the user-item interaction matrix.
        """
        self.user_item_matrix = user_item_matrix
        self.matrix = self.svd.fit_transform(user_item_matrix)
        return self.matrix
        
    def recommend(self, user_index, top_n=5):
        """
        Recommends items for a user based on cosine similarity in latent space.
        """
        if self.matrix is None:
            return []
            
        user_vector = self.matrix[user_index].reshape(1, -1)
        # Simple similarity: dot product of latent vectors
        scores = np.dot(self.matrix, user_vector.T).flatten()
        
        # Sort and get top N excluding the user themselves
        recommended_indices = np.argsort(scores)[::-1]
        recommended_indices = [i for i in recommended_indices if i != user_index][:top_n]
        
        return recommended_indices

def get_engine(n_components=10):
    return RecommendationEngine(n_components=n_components)
