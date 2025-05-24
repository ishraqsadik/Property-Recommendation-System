import numpy as np
import pandas as pd
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
from datetime import datetime

from app.ml.model import PropertyRecommendationModel
from app.ml.feature_engineering import FEATURES, prepare_training_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackLearningSystem:
    """
    System that learns from human feedback to improve the property recommendation model
    """
    
    def __init__(
        self,
        model: PropertyRecommendationModel,
        feedback_db_path: str = 'data/feedback/feedback_database.json',
        training_data_path: str = 'data/processed/training_data.pkl',
        model_save_path: str = 'data/models/recommendation_model.pkl',
        retrain_threshold: int = 10
    ):
        """
        Initialize the feedback learning system
        
        Args:
            model: Trained PropertyRecommendationModel instance
            feedback_db_path: Path to store feedback
            training_data_path: Path to store/load training data
            model_save_path: Path to save updated model
            retrain_threshold: Number of feedback items needed to trigger retraining
        """
        self.model = model
        self.feedback_db_path = feedback_db_path
        self.training_data_path = training_data_path
        self.model_save_path = model_save_path
        self.retrain_threshold = retrain_threshold
        
        # Initialize or load feedback database
        self.feedback_db = self._load_feedback_db()
        
        # Load original training data if available
        self.original_training_data = None
        if os.path.exists(training_data_path):
            try:
                self.original_training_data = pd.read_pickle(training_data_path)
                logger.info(f"Loaded original training data from {training_data_path}")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
    
    def _load_feedback_db(self) -> Dict:
        """
        Load feedback database from disk
        
        Returns:
            Dict containing feedback data
        """
        if os.path.exists(self.feedback_db_path):
            try:
                with open(self.feedback_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback database: {e}")
        
        # Create new feedback database if it doesn't exist
        feedback_db = {
            'feedbacks': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'last_retrain': None
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.feedback_db_path), exist_ok=True)
        
        # Save empty database
        with open(self.feedback_db_path, 'w') as f:
            json.dump(feedback_db, f, indent=2)
        
        return feedback_db
    
    def _save_feedback_db(self) -> None:
        """
        Save feedback database to disk
        """
        # Update metadata
        self.feedback_db['metadata']['last_updated'] = datetime.now().isoformat()
        self.feedback_db['metadata']['total_count'] = len(self.feedback_db['feedbacks'])
        self.feedback_db['metadata']['positive_count'] = sum(
            1 for f in self.feedback_db['feedbacks'] if f['is_approved']
        )
        self.feedback_db['metadata']['negative_count'] = sum(
            1 for f in self.feedback_db['feedbacks'] if not f['is_approved']
        )
        
        # Save to disk
        with open(self.feedback_db_path, 'w') as f:
            json.dump(self.feedback_db, f, indent=2)
        
        logger.info(f"Feedback database saved with {self.feedback_db['metadata']['total_count']} entries")
    
    def add_feedback(
        self,
        subject_id: str,
        subject_data: Dict[str, Any],
        recommended_properties: List[Dict[str, Any]],
        is_approved: bool,
        selected_properties: Optional[List[Dict[str, Any]]] = None,
        comments: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add user feedback to the database
        
        Args:
            subject_id: ID of the subject property
            subject_data: Data about the subject property
            recommended_properties: List of properties recommended by the model
            is_approved: Whether the appraiser approved the recommendations
            selected_properties: Alternative properties selected by the appraiser
            comments: Optional textual feedback
            
        Returns:
            Dict with feedback ID and status
        """
        feedback_id = f"feedback_{len(self.feedback_db['feedbacks']) + 1}"
        
        feedback_entry = {
            'id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'subject_id': subject_id,
            'subject_data': subject_data,
            'recommended_properties': recommended_properties,
            'is_approved': is_approved,
            'selected_properties': selected_properties or [],
            'comments': comments or '',
            'processed': False
        }
        
        # Add to database
        self.feedback_db['feedbacks'].append(feedback_entry)
        
        # Save updated database
        self._save_feedback_db()
        
        # Check if we should retrain
        new_feedback_count = sum(1 for f in self.feedback_db['feedbacks'] if not f['processed'])
        if new_feedback_count >= self.retrain_threshold:
            logger.info(f"Retraining threshold reached ({new_feedback_count} new feedbacks)")
            self.retrain_model()
        
        return {
            'feedback_id': feedback_id,
            'status': 'success',
            'new_feedback_count': new_feedback_count,
            'retrain_threshold': self.retrain_threshold
        }
    
    def retrain_model(self, force: bool = False) -> Dict[str, Any]:
        """
        Retrain the model with accumulated feedback
        
        Args:
            force: Whether to force retraining even if threshold not reached
            
        Returns:
            Dict with retraining results
        """
        # Count unprocessed feedback
        new_feedback_count = sum(1 for f in self.feedback_db['feedbacks'] if not f['processed'])
        
        if new_feedback_count < self.retrain_threshold and not force:
            logger.info(f"Not enough new feedback ({new_feedback_count}/{self.retrain_threshold}) to trigger retraining")
            return {
                'status': 'skipped',
                'reason': 'Not enough new feedback',
                'new_feedback_count': new_feedback_count,
                'retrain_threshold': self.retrain_threshold
            }
        
        logger.info(f"Retraining model with {new_feedback_count} new feedback entries")
        
        # Store existing test property IDs before retraining
        existing_test_property_ids = self.model.test_property_ids.copy()
        
        # Prepare feedback data for training
        feedback_data = self._prepare_feedback_data()
        
        # Check if we have any feedback data to work with
        if feedback_data['X'].empty and (self.original_training_data is None or self.original_training_data['X'].empty):
            logger.warning("No training data available (neither feedback nor original)")
            return {
                'status': 'skipped',
                'reason': 'No training data available',
                'processed_feedback_count': 0
            }
        
        # If we have original training data, combine it with feedback data
        if self.original_training_data is not None and not self.original_training_data['X'].empty:
            X_combined = pd.concat([self.original_training_data['X'], feedback_data['X']], ignore_index=True)
            y_combined = pd.concat([self.original_training_data['y'], feedback_data['y']], ignore_index=True)
            
            # Group information for cross-validation
            if 'groups' in self.original_training_data and not feedback_data['groups'].empty:
                groups_combined = pd.concat([
                    self.original_training_data['groups'], 
                    feedback_data['groups']
                ], ignore_index=True)
            elif 'groups' in self.original_training_data:
                groups_combined = self.original_training_data['groups']
            elif not feedback_data['groups'].empty:
                groups_combined = feedback_data['groups']
            else:
                groups_combined = None
        elif not feedback_data['X'].empty:
            # Only feedback data available
            X_combined = feedback_data['X']
            y_combined = feedback_data['y']
            groups_combined = feedback_data['groups'] if not feedback_data['groups'].empty else None
        else:
            # This shouldn't happen due to the earlier check, but just in case
            logger.error("No valid training data found")
            return {
                'status': 'failed',
                'reason': 'No valid training data found',
                'processed_feedback_count': 0
            }
        
        # Retrain the model
        metrics = self.model.fit(X_combined, y_combined, groups=groups_combined)
        
        # Merge the existing test property IDs with new ones to ensure we don't lose test properties
        if existing_test_property_ids:
            logger.info(f"Preserving {len(existing_test_property_ids)} existing test property IDs")
            self.model.test_property_ids.update(existing_test_property_ids)
            logger.info(f"Model now has {len(self.model.test_property_ids)} test property IDs for validation")
        
        # Save the updated model
        self.model.save_model(self.model_save_path)
        
        # Mark feedback as processed
        for feedback in self.feedback_db['feedbacks']:
            if not feedback['processed']:
                feedback['processed'] = True
        
        # Update last retrain timestamp
        self.feedback_db['metadata']['last_retrain'] = datetime.now().isoformat()
        
        # Save updated database
        self._save_feedback_db()
        
        logger.info(f"Model retrained and saved to {self.model_save_path}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'processed_feedback_count': new_feedback_count
        }
    
    def _prepare_feedback_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data from feedback
        
        Returns:
            Dict with X (features), y (labels), and groups
        """
        # Collect positive examples (approved or manually selected properties)
        positive_rows = []
        
        # Collect negative examples (rejected properties)
        negative_rows = []
        
        # Process each feedback entry
        for feedback in self.feedback_db['feedbacks']:
            if feedback['processed']:
                continue
                
            subject_id = feedback['subject_id']
            
            if feedback['is_approved']:
                # If recommendations were approved, add them as positive examples
                for prop in feedback['recommended_properties']:
                    row = {feature: prop.get(feature, np.nan) for feature in FEATURES}
                    row['is_selected'] = 1
                    row['order_id'] = subject_id
                    positive_rows.append(row)
            elif feedback['selected_properties']:
                # If alternatives were provided, add them as positive examples
                for prop in feedback['selected_properties']:
                    row = {feature: prop.get(feature, np.nan) for feature in FEATURES}
                    row['is_selected'] = 1
                    row['order_id'] = subject_id
                    positive_rows.append(row)
                    
                # Add model recommendations as negative examples
                for prop in feedback['recommended_properties']:
                    # Check if this property was also manually selected
                    prop_address = prop.get('address', '').lower().strip()
                    if not any(p.get('address', '').lower().strip() == prop_address 
                              for p in feedback['selected_properties']):
                        row = {feature: prop.get(feature, np.nan) for feature in FEATURES}
                        row['is_selected'] = 0
                        row['order_id'] = subject_id
                        negative_rows.append(row)
            else:
                # If recommendations were rejected without alternatives, 
                # add them as weak negative examples
                for prop in feedback['recommended_properties']:
                    row = {feature: prop.get(feature, np.nan) for feature in FEATURES}
                    row['is_selected'] = 0
                    row['order_id'] = subject_id
                    negative_rows.append(row)
        
        # Create DataFrames
        df_positives = pd.DataFrame(positive_rows)
        df_negatives = pd.DataFrame(negative_rows)
        
        # Combine positives and negatives
        df_combined = pd.concat([df_positives, df_negatives], ignore_index=True)
        
        if df_combined.empty:
            logger.warning("No feedback data to process")
            return {
                'X': pd.DataFrame(columns=FEATURES), 
                'y': pd.Series(dtype=int),
                'groups': pd.Series(dtype=str)  # Add empty groups series
            }
        
        # Extract features, labels, and groups
        X = df_combined[FEATURES]
        y = df_combined['is_selected']
        groups = df_combined['order_id']
        
        return {'X': X, 'y': y, 'groups': groups}
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback
        
        Returns:
            Dict with feedback statistics
        """
        total = len(self.feedback_db['feedbacks'])
        
        if total == 0:
            return {
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'processed_count': 0,
                'unprocessed_count': 0,
                'approval_rate': 0.0,
                'last_updated': self.feedback_db['metadata']['last_updated'],
                'last_retrain': self.feedback_db['metadata']['last_retrain']
            }
        
        positive_count = sum(1 for f in self.feedback_db['feedbacks'] if f['is_approved'])
        processed_count = sum(1 for f in self.feedback_db['feedbacks'] if f['processed'])
        
        return {
            'total_count': total,
            'positive_count': positive_count,
            'negative_count': total - positive_count,
            'processed_count': processed_count,
            'unprocessed_count': total - processed_count,
            'approval_rate': positive_count / total if total > 0 else 0.0,
            'last_updated': self.feedback_db['metadata']['last_updated'],
            'last_retrain': self.feedback_db['metadata']['last_retrain']
        }
    
    def save_training_data(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> None:
        """
        Save original training data for future retraining
        
        Args:
            X: Feature DataFrame
            y: Target Series
            groups: Optional group IDs for cross-validation
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        
        # Create dictionary with training data
        training_data = {'X': X, 'y': y}
        if groups is not None:
            training_data['groups'] = groups
        
        # Save to disk
        pd.to_pickle(training_data, self.training_data_path)
        
        # Update instance variable
        self.original_training_data = training_data
        
        logger.info(f"Original training data saved to {self.training_data_path}")
    
    def export_feedback_to_csv(self, path: str) -> str:
        """
        Export feedback database to CSV for analysis
        
        Args:
            path: Path to save the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Flatten feedback data for CSV
        rows = []
        for feedback in self.feedback_db['feedbacks']:
            base_row = {
                'feedback_id': feedback['id'],
                'timestamp': feedback['timestamp'],
                'subject_id': feedback['subject_id'],
                'is_approved': feedback['is_approved'],
                'comments': feedback['comments'],
                'processed': feedback['processed']
            }
            
            # Add subject data
            for key, value in feedback['subject_data'].items():
                base_row[f'subject_{key}'] = value
            
            # Add rows for recommended properties
            for i, prop in enumerate(feedback['recommended_properties']):
                row = base_row.copy()
                row['property_type'] = 'recommended'
                row['property_index'] = i
                for key, value in prop.items():
                    row[f'property_{key}'] = value
                rows.append(row)
            
            # Add rows for selected properties (if any)
            for i, prop in enumerate(feedback['selected_properties']):
                row = base_row.copy()
                row['property_type'] = 'selected'
                row['property_index'] = i
                for key, value in prop.items():
                    row[f'property_{key}'] = value
                rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        
        logger.info(f"Feedback data exported to {path}")
        
        return path 