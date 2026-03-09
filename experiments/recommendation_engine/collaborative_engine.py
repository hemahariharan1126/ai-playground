import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeRecommender:
    def __init__(self, n_components=10):
        self.svd = TruncatedSVD(n_components=n_components)
        self.user_item_matrix = None
        self.latent_matrix = None

    def fit(self, matrix):
        """
        Fits SVD to the User-Item matrix.
        """
        self.user_item_matrix = matrix
        self.latent_matrix = self.svd.fit_transform(matrix)
        return self.latent_matrix

    def recommend(self, user_index, top_n=5):
        """
        Recommends items for a user based on their latent vector similarity to items.
        """
        if self.latent_matrix is None:
            return []

        # User vector in latent space
        user_vec = self.latent_matrix[user_index].reshape(1, -1)
        
        # Item vectors in latent space (V^T matrix)
        # SVD: R ≈ U * Sigma * V^T, items are represented by rows of V (or columns of U*Sigma if looking at user latent)
        # For simplicity, we'll use the reconstructed ratings
        reconstructed_ratings = self.svd.inverse_transform(user_vec).flatten()
        
        # Sort ratings and filter out items user already interacted with
        item_scores = list(enumerate(reconstructed_ratings))
        # Filter items where user rating was non-zero (simple interaction proxy)
        interacted_items = np.where(self.user_item_matrix[user_index] > 0)[0]
        
        recommendations = sorted([i for i in item_scores if i[0] not in interacted_items], 
                                  key=lambda x: x[1], reverse=True)[:top_n]
        
        return [i[0] for i in recommendations]

if __name__ == "__main__":
    # 5 Users, 10 Items
    np.random.seed(42)
    data = np.random.randint(0, 6, size=(5, 10))
    
    recommender = CollaborativeRecommender(n_components=3)
    recommender.fit(data)
    
    # Recommend for User 0
    recs = recommender.recommend(0, top_n=3)
    print(f"Recommendations for User 0: Items {recs}")
