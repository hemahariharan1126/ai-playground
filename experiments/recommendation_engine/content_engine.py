import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.df = None

    def fit(self, df, text_column='description'):
        """
        Fits the TF-IDF model to the item metadata.
        """
        self.df = df.reset_index(drop=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df[text_column])
        return self.tfidf_matrix

    def recommend(self, item_index, top_n=5):
        """
        Recommends items similar to the given item based on content.
        """
        if self.tfidf_matrix is None:
            return []

        # Compute cosine similarity between the item and all others
        cosine_sim = cosine_similarity(self.tfidf_matrix[item_index], self.tfidf_matrix).flatten()
        
        # Sort by similarity score (descending)
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N items (excluding the item itself)
        top_indices = [i[0] for i in sim_scores if i[0] != item_index][:top_n]
        
        return self.df.iloc[top_indices]

def get_content_recommender():
    return ContentRecommender()

if __name__ == "__main__":
    # Sample Movie Dataset
    data = {
        'title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'Interstellar', 'The Dark Knight'],
        'description': [
            'A computer hacker learns about the true nature of reality.',
            'A thief who enters the dreams of others to steal secrets.',
            'A cowboy doll and a space ranger go on an adventure.',
            'A father fish searches for his lost son in the ocean.',
            'Astronauts travel through a wormhole to find a new home.',
            'A vigilante fights crime in Gotham City against the Joker.'
        ]
    }
    df = pd.DataFrame(data)
    
    recommender = get_content_recommender()
    recommender.fit(df)
    
    # Recommend movies similar to 'Inception' (index 1)
    results = recommender.recommend(1, top_n=2)
    print("Recommendations for 'Inception':")
    print(results['title'].values)
