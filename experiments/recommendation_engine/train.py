import numpy as np
import os
import sys
import mlflow

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.recommendation_engine.model import get_engine
from utils.logger import logger

def run_demo():
    logger.info("Initializing Recommendation Engine Demo...")
    
    # Create a synthetic User-Item Matrix (e.g., Users x Movies)
    # Rows: 10 Users, Columns: 20 Items
    np.random.seed(42)
    user_item_matrix = np.random.randint(0, 6, size=(10, 20))
    
    logger.info("Sample User-Item Interaction Matrix (User 0):")
    logger.info(user_item_matrix[0])
    
    engine = get_engine(n_components=5)
    
    mlflow.set_experiment("Recommendation_Engine")
    with mlflow.start_run():
        mlflow.log_param("n_components", 5)
        engine.fit(user_item_matrix)
        mlflow.log_metric("fit_complete", 1.0)
    
    user_id = 0
    recommendations = engine.recommend(user_id, top_n=3)
    
    logger.info(f"Top 3 recommendations for User {user_id}: Users {recommendations}")
    logger.info("Note: In a real scenario, these would map to specific items/movies.")

if __name__ == "__main__":
    run_demo()
