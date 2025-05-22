import os
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to disk
    
    Args:
        df: DataFrame to save
        path: Path to save the DataFrame
    """
    ensure_dir(os.path.dirname(path))
    
    # Determine file format based on extension
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        df.to_pickle(path)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load DataFrame from disk
    
    Args:
        path: Path to load the DataFrame from
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Determine file format based on extension
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        return pd.read_pickle(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def save_model(model: Any, path: str) -> None:
    """
    Save model to disk
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str) -> Any:
    """
    Load model from disk
    
    Args:
        path: Path to load the model from
        
    Returns:
        Loaded model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def property_to_features(property_data: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
    """
    Extract features from property data
    
    Args:
        property_data: Dictionary with property data
        feature_list: List of feature names to extract
        
    Returns:
        Dictionary with extracted features
    """
    features = {}
    
    for feature in feature_list:
        features[feature] = property_data.get(feature, np.nan)
    
    return features

def format_property_record(prop_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format property data for display
    
    Args:
        prop_data: Raw property data
        
    Returns:
        Formatted property data
    """
    formatted = {}
    
    # Basic information - ensure no NaN values for string fields
    formatted['address'] = None if pd.isna(prop_data.get('address')) else str(prop_data.get('address', 'Unknown'))
    formatted['city'] = None if pd.isna(prop_data.get('city')) else str(prop_data.get('city', ''))
    formatted['province'] = None if pd.isna(prop_data.get('province')) else str(prop_data.get('province', ''))
    
    # Handle postal_code specially to convert NaN to None
    postal_code = prop_data.get('postal_code', '')
    formatted['postal_code'] = None if pd.isna(postal_code) else str(postal_code)
    
    # Property details
    formatted['structure_type'] = None if pd.isna(prop_data.get('structure_type')) else str(prop_data.get('structure_type', 'Unknown'))
    
    # Handle numeric fields - convert NaN to None
    for field in ['bedrooms', 'gla', 'age', 'sale_price', 'room_count', 'full_baths', 'half_baths', 'lot_size', 'year_built']:
        value = prop_data.get(field, np.nan)
        formatted[field] = None if pd.isna(value) else value
    
    # Handle other string fields to prevent NaN values
    for field in ['stories', 'basement', 'heating', 'cooling', 'public_remarks']:
        value = prop_data.get(field, '')
        formatted[field] = None if pd.isna(value) else str(value)
        
    # Handle latitude/longitude
    for field in ['latitude', 'longitude']:
        value = prop_data.get(field, np.nan)
        formatted[field] = None if pd.isna(value) else float(value)
    
    # Format sale date
    sale_date = prop_data.get('sale_date')
    if sale_date is not None and not pd.isna(sale_date):
        if isinstance(sale_date, pd.Timestamp):
            formatted['sale_date'] = sale_date.strftime('%Y-%m-%d')
        else:
            formatted['sale_date'] = str(sale_date)
    else:
        formatted['sale_date'] = None
    
    # Model score if available
    if 'score' in prop_data and not pd.isna(prop_data['score']):
        formatted['score'] = round(float(prop_data['score']), 3)
    
    # Ensure ID and order_id are included and not NaN
    formatted['id'] = None if pd.isna(prop_data.get('id')) else str(prop_data.get('id', ''))
    formatted['order_id'] = None if pd.isna(prop_data.get('order_id')) else str(prop_data.get('order_id', ''))
    
    # Ensure is_true_comp is properly formatted
    if 'is_true_comp' in prop_data:
        formatted['is_true_comp'] = int(prop_data.get('is_true_comp', 0))
    
    return formatted

def format_currency(value: Optional[float], currency: str = '$') -> str:
    """
    Format a number as currency
    
    Args:
        value: Number to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return 'N/A'
    
    return f"{currency}{value:,.0f}"

def calculate_metrics(actual: List[str], predicted: List[str]) -> Dict[str, float]:
    """
    Calculate performance metrics for recommendations
    
    Args:
        actual: List of actual property IDs
        predicted: List of predicted property IDs
        
    Returns:
        Dictionary with performance metrics
    """
    actual_set = set(actual)
    predicted_set = set(predicted)
    
    # Calculate intersection
    intersection = actual_set.intersection(predicted_set)
    
    # Calculate metrics
    precision = len(intersection) / len(predicted_set) if predicted_set else 0
    recall = len(intersection) / len(actual_set) if actual_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_matches': len(intersection),
        'num_actual': len(actual_set),
        'num_predicted': len(predicted_set)
    }

def log_prediction(
    subject_id: str,
    subject_data: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    log_path: str = 'data/logs/predictions.jsonl'
) -> None:
    """
    Log a prediction to a JSONL file
    
    Args:
        subject_id: ID of the subject property
        subject_data: Data about the subject property
        predictions: List of predicted properties
        log_path: Path to the log file
    """
    ensure_dir(os.path.dirname(log_path))
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'subject_id': subject_id,
        'subject_data': subject_data,
        'predictions': predictions
    }
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n') 