import logging
import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import json

from app.ml.data_processing import load_and_process_data, deduplicate_properties, handle_missing_values
from app.ml.feature_engineering import prepare_training_data
from app.ml.model import PropertyRecommendationModel
from app.ml.feedback_learning import FeedbackLearningSystem

# Configure logging
logger = logging.getLogger(__name__)

def load_data_from_notebook(json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process data from the original JSON file used in the notebook
    
    Args:
        json_path: Path to the JSON data file
        
    Returns:
        Tuple of (subjects_df, comps_df, properties_df)
    """
    logger.info(f"Loading data from {json_path}")
    subjects_df, comps_df, properties_df = load_and_process_data(json_path)
    
    # Deduplicate properties
    properties_df = deduplicate_properties(properties_df)
    
    # Handle missing values
    handle_missing_values(subjects_df, comps_df, properties_df)
    
    # Ensure consistent ID format - convert to string
    if 'id' in properties_df.columns:
        properties_df['id'] = properties_df['id'].astype(str)
        logger.info("Converted property IDs to string format for consistency")
    
    return subjects_df, comps_df, properties_df

def initialize_model_from_data(
    json_path: str,
    model_path: str = "data/models/recommendation_model.pkl",
    force_retrain: bool = False
) -> PropertyRecommendationModel:
    """
    Initialize a model from the data, training if necessary
    
    Args:
        json_path: Path to the JSON data file
        model_path: Path to save/load the model
        force_retrain: Whether to force retraining even if model exists
        
    Returns:
        Trained PropertyRecommendationModel
    """
    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        logger.info(f"Loading existing model from {model_path}")
        model = PropertyRecommendationModel(model_path=model_path)
        return model
    
    # Load and process data
    logger.info("Loading and processing data")
    subjects_df, comps_df, properties_df = load_data_from_notebook(json_path)
    
    # Ensure properties DataFrame has string IDs for consistency
    if 'id' in properties_df.columns:
        properties_df['id'] = properties_df['id'].astype(str)
        logger.info(f"Ensuring property IDs are strings for consistency: {properties_df['id'].dtype}")
    
    # Prepare training data
    logger.info("Preparing training data")
    X, y, groups = prepare_training_data(comps_df, properties_df)
    
    # Create and train model
    logger.info("Training model")
    model = PropertyRecommendationModel()
    model.fit(X, y, groups=groups)
    
    # Ensure test_property_ids are all strings
    model.test_property_ids = {str(id) for id in model.test_property_ids}
    logger.info(f"Model has {len(model.test_property_ids)} test property IDs, all converted to string format")
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    model.save_model(model_path)
    
    # Setup feedback system with original training data
    feedback_system = FeedbackLearningSystem(
        model=model,
        model_save_path=model_path
    )
    feedback_system.save_training_data(X, y, groups)
    
    return model

def initialize_feedback_system(
    model: PropertyRecommendationModel,
    feedback_db_path: str = "data/feedback/feedback_database.json",
    training_data_path: str = "data/processed/training_data.pkl",
    model_save_path: str = "data/models/recommendation_model.pkl",
    retrain_threshold: int = 10
) -> FeedbackLearningSystem:
    """
    Initialize the feedback learning system
    
    Args:
        model: Trained model
        feedback_db_path: Path to feedback database
        training_data_path: Path to original training data
        model_save_path: Path to save/load the model
        retrain_threshold: Number of feedbacks needed to trigger retraining
        
    Returns:
        FeedbackLearningSystem instance
    """
    return FeedbackLearningSystem(
        model=model,
        feedback_db_path=feedback_db_path,
        training_data_path=training_data_path,
        model_save_path=model_save_path,
        retrain_threshold=retrain_threshold
    )

__all__ = [
    'PropertyRecommendationModel',
    'FeedbackLearningSystem',
    'load_data_from_notebook',
    'initialize_model_from_data',
    'initialize_feedback_system'
] 