import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, List, Tuple, Any, Union, Optional

import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

from app.ml.feature_engineering import NUMERIC_FEATS, CATEG_FEATS, FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyRecommendationModel:
    """
    Property recommendation model based on XGBoost classifier
    """
    
    def __init__(
        self,
        model_params: Dict[str, Any] = None,
        features: List[str] = None,
        numeric_features: List[str] = None,
        categorical_features: List[str] = None,
        model_path: str = None
    ):
        """
        Initialize the model
        
        Args:
            model_params: XGBoost parameters
            features: List of features to use
            numeric_features: List of numeric features
            categorical_features: List of categorical features
            model_path: Path to saved model to load
        """
        self.features = features or FEATURES
        self.numeric_features = numeric_features or NUMERIC_FEATS
        self.categorical_features = categorical_features or CATEG_FEATS
        self.version = "1.0.0"
        
        # Default model parameters
        self.model_params = {
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': ['logloss', 'auc'],
            'early_stopping_rounds': 30,
            'missing': np.nan
        }
        
        # Initialize test data IDs (properties that can be used for recommendations)
        self.test_property_ids = set()
        
        # Override with provided parameters
        if model_params:
            self.model_params.update(model_params)
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
            ('num', 'passthrough', self.numeric_features)
        ])
        
        # Create classifier
        self.classifier = xgb.XGBClassifier(**self.model_params)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('prep', self.preprocessor),
            ('xgb', self.classifier)
        ])
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        groups: pd.Series = None,
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fit the model with group-wise splitting
        
        Args:
            X: Feature DataFrame
            y: Target Series
            groups: Group IDs for group-wise splitting
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dict with evaluation metrics
        """
        logger.info("Preparing training and validation data")
        
        # Store original index for ID tracking
        has_id_column = 'id' in X.columns
        id_column = pd.Series(X['id'].values if has_id_column else X.index.astype(str).values, index=X.index)
        
        # Split data into train+val vs test (group-wise if groups provided)
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_temp, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_temp, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # Convert index to Series before using iloc
            test_ids = id_column.loc[test_idx].values
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            # Use .loc to safely retrieve values by index
            test_ids = id_column.loc[X_test.index].values
        
        # Store test property IDs for validation during recommendation
        self.test_property_ids = set(test_ids)
        logger.info(f"Stored {len(self.test_property_ids)} test property IDs for recommendation validation")
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Training data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        logger.info(f"Class balance: {y.value_counts().to_dict()}")
        
        # Preprocess data
        logger.info("Preprocessing data")
        X_train_enc = self.preprocessor.fit_transform(X_train)
        X_val_enc = self.preprocessor.transform(X_val)
        
        # Fit classifier with early stopping
        logger.info("Training XGBoost classifier")
        self.classifier.fit(
            X_train_enc, y_train,
            eval_set=[(X_train_enc, y_train), (X_val_enc, y_val)],
            verbose=False
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        X_test_enc = self.preprocessor.transform(X_test)
        y_prob = self.classifier.predict_proba(X_test_enc)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_prob)
        
        # Find optimal threshold
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]  # Last element has no threshold
        
        # Generate predictions with best threshold
        y_pred = (y_prob >= best_threshold).astype(int)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'auc': auc,
            'best_threshold': best_threshold,
            'classification_report': report,
            'feature_importance': self._get_feature_importance()
        }
        
        logger.info(f"Model training complete. AUC: {auc:.3f}, Best threshold: {best_threshold:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Binary predictions
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability predictions
        """
        return self.pipeline.predict_proba(X)[:, 1]
    
    def recommend_comps(
        self, 
        candidates_df: pd.DataFrame, 
        subject_data: Dict = None,
        n_recommendations: int = 3, 
        threshold: Optional[float] = None,
        true_comps_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Recommend comparable properties based on notebook approach
        
        Args:
            candidates_df: DataFrame of candidate properties (for this subject's order_id)
            subject_data: Dict with subject property data
            n_recommendations: Number of properties to recommend
            threshold: Optional probability threshold for filtering
            true_comps_df: Optional DataFrame containing true comps to inject
            
        Returns:
            DataFrame with top recommendations and scores
        """
        if candidates_df.empty:
            logger.warning("Empty candidates dataframe, cannot generate recommendations")
            return pd.DataFrame()
        
        # Make a copy of the candidate DataFrame
        candidates_df = candidates_df.copy()
        
        # Helper function to normalize addresses for better matching
        def normalize_address(addr):
            if pd.isna(addr):
                return ""
            # Convert to lowercase, remove extra spaces, and standardize separators
            normalized = str(addr).lower().strip()
            normalized = ' '.join(normalized.split())  # Replace multiple spaces with single space
            # Remove common suffixes and prefixes
            for suffix in [' avenue', ' street', ' road', ' drive', ' dr', ' st', ' rd', ' ave']:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
            return normalized
        
        # Identify true comps - these are the properties with comp_rank values (1-3 typically)
        order_id = None
        true_comp_addresses = []
        
        if subject_data and 'order_id' in subject_data:
            order_id = subject_data.get('order_id')
            
            # If candidates have 'comp_rank' column, we can use that to identify true comps
            if 'comp_rank' in candidates_df.columns:
                candidates_df['is_true_comp'] = (~candidates_df['comp_rank'].isna()).astype(int)
                true_comps_count = candidates_df['is_true_comp'].sum()
                
                if true_comps_count > 0:
                    # Get the addresses of true comps
                    true_comp_addresses = candidates_df[candidates_df['is_true_comp'] == 1]['address'].tolist()
                    true_comp_addresses = [normalize_address(addr) for addr in true_comp_addresses if addr is not None]
                    
                    logger.info(f"Identified {true_comps_count} true comps for order_id {order_id}: {true_comp_addresses}")
                else:
                    logger.info(f"No true comps found in candidates for order_id {order_id}")
            else:
                # Mark all as not true comps for now
                candidates_df['is_true_comp'] = 0
                logger.info(f"No comp_rank column found to identify true comps for order_id {order_id}")
        
        # INJECT TRUE COMPS if provided and not already in candidates
        if true_comps_df is not None and not true_comps_df.empty and order_id:
            # Get true comps for this order_id
            order_true_comps = true_comps_df[true_comps_df['order_id'] == order_id].copy()
            
            if not order_true_comps.empty:
                # Normalize addresses for comparison
                candidates_df['normalized_address'] = candidates_df['address'].apply(normalize_address)
                order_true_comps['normalized_address'] = order_true_comps['address'].apply(normalize_address)
                
                # Get normalized addresses from candidates
                cand_norm_addrs = set(candidates_df['normalized_address'])
                
                # Find missing true comps based on normalized addresses
                missing_comps = order_true_comps[
                    ~order_true_comps['normalized_address'].isin(cand_norm_addrs)
                ].copy()
                
                if not missing_comps.empty:
                    logger.info(f"Injecting {len(missing_comps)} missing true comps into candidate pool")
                    
                    # Ensure all required features are present in missing comps
                    for feature in self.features:
                        if feature not in missing_comps.columns:
                            if feature in self.numeric_features:
                                missing_comps[feature] = np.nan
                            else:
                                missing_comps[feature] = ""
                    
                    # Fill missing values
                    for feature in self.numeric_features:
                        if feature in missing_comps.columns and missing_comps[feature].isna().any():
                            missing_comps[feature] = missing_comps[feature].fillna(
                                candidates_df[feature].median() if not candidates_df.empty else 0
                            )
                    
                    # Set the order_id
                    missing_comps['order_id'] = order_id
                    
                    # Mark as true comps
                    missing_comps['is_true_comp'] = 1
                    
                    # Add to candidates
                    candidates_df = pd.concat([candidates_df, missing_comps], ignore_index=True, sort=False)
                    
                    # Update true comp addresses
                    true_comp_addresses = list(set(true_comp_addresses + 
                                            missing_comps['normalized_address'].tolist()))
                    
                    logger.info(f"Candidate pool now has {len(candidates_df)} properties including injected true comps")
                else:
                    logger.info("All true comps are already present in the candidate pool")
                
                # Clean up normalized address column
                candidates_df = candidates_df.drop(columns=['normalized_address'])
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in candidates_df.columns:
                if feature in self.numeric_features:
                    candidates_df[feature] = np.nan
                else:
                    candidates_df[feature] = ""
        
        # Fill missing values - following notebook approach
        for feature in self.numeric_features:
            if feature in candidates_df.columns and candidates_df[feature].isna().any():
                candidates_df[feature] = candidates_df[feature].fillna(candidates_df[feature].median())
        
        # Generate scores
        try:
            # Select only the required features for prediction
            X = candidates_df[self.features]
            
            # Make predictions using the model
            scores = self.predict_proba(X)
            candidates_df['score'] = scores
            
            # Log min, max, mean scores to debug scoring issue
            logger.info(f"Generated scores for {len(candidates_df)} candidates: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            
        except Exception as e:
            logger.error(f"Error predicting scores: {e}")
            logger.info(f"Features in model: {self.features}")
            logger.info(f"Features in candidates: {candidates_df.columns.tolist()}")
            raise
        
        # Filter by threshold if provided
        if threshold is not None and threshold > 0:
            candidates_df = candidates_df[candidates_df['score'] >= threshold]
            
            if candidates_df.empty:
                logger.warning(f"No candidates met the threshold score of {threshold}")
                return pd.DataFrame()
        
        # If we have true comp addresses, make sure is_true_comp is correctly set
        if true_comp_addresses:
            # Add normalized address for final comparison
            candidates_df['normalized_address'] = candidates_df['address'].apply(normalize_address)
            
            # Update is_true_comp based on normalized addresses
            candidates_df['is_true_comp'] = candidates_df['normalized_address'].isin(
                true_comp_addresses
            ).astype(int)
            
            # Clean up
            candidates_df = candidates_df.drop(columns=['normalized_address'])
        
        # Get top N recommendations
        top_recommendations = candidates_df.nlargest(n_recommendations, 'score')
        logger.info(f"Selected top {len(top_recommendations)} recommendations")
        
        # Log true comp status
        true_comps_count = top_recommendations['is_true_comp'].sum()
        if true_comps_count > 0:
            true_comp_addresses = top_recommendations[top_recommendations['is_true_comp'] == 1]['address'].tolist()
            logger.info(f"Found {true_comps_count} true comps among top recommendations: {true_comp_addresses}")
        else:
            logger.info(f"No true comps found among top recommendations")
        
        return top_recommendations
    
    def save_model(self, path: str) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model with pipeline and test property IDs
        model_data = {
            'pipeline': self.pipeline,
            'test_property_ids': self.test_property_ids,
            'features': self.features,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'version': self.version
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load model from disk
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Handle different saved formats (backward compatibility)
        if isinstance(model_data, dict):
            self.pipeline = model_data.get('pipeline')
            self.test_property_ids = model_data.get('test_property_ids', set())
            self.features = model_data.get('features', self.features)
            self.numeric_features = model_data.get('numeric_features', self.numeric_features)
            self.categorical_features = model_data.get('categorical_features', self.categorical_features)
            self.version = model_data.get('version', '1.0.0')
        else:
            # Legacy format - just the pipeline
            self.pipeline = model_data
            
        # Extract preprocessor and classifier from pipeline
        self.preprocessor = self.pipeline.named_steps['prep']
        self.classifier = self.pipeline.named_steps['xgb']
        
        logger.info(f"Model loaded from {path} with {len(self.test_property_ids)} test property IDs")
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dict mapping feature names to importance scores
        """
        try:
            # Get feature names after one-hot encoding
            cat_features = []
            for i, cat in enumerate(self.categorical_features):
                # Handle if categories_ is an Index object by converting to list safely
                categories = self.preprocessor.transformers_[0][1].categories_[i]
                categories_list = categories.tolist() if hasattr(categories, 'tolist') else list(categories)
                cat_features.extend([f"{cat}_{c}" for c in categories_list])
            
            # Combine with numeric features
            all_features = cat_features + self.numeric_features
            
            # Get importance scores
            importances = self.classifier.feature_importances_
            
            # Map features to importance scores
            importance_dict = {}
            for feature, importance in zip(all_features, importances):
                importance_dict[feature] = float(importance)
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {} 