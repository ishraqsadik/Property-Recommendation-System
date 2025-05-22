import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define features used in the model - exactly matching the notebook
NUMERIC_FEATS = [
    'gla',
    'full_baths',
    'age',
    'bedrooms',
    'room_count',
    'sale_price'  # Adding sale_price back as in notebook
]
CATEG_FEATS = [
    'structure_type', 
    'stories'
]
FEATURES = NUMERIC_FEATS + CATEG_FEATS

def prepare_training_data(
    comps_df: pd.DataFrame,
    properties_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training data for the recommendation model
    
    Args:
        comps_df: DataFrame of comparable properties
        properties_df: DataFrame of properties
        
    Returns:
        Tuple of (X, y, groups) where:
            X: Feature DataFrame
            y: Target Series (1 for selected comp, 0 otherwise)
            groups: Group IDs for group-wise CV (order_id)
    """
    logger.info("Starting training data preparation")
    
    # Create a copy of the dataframes to avoid modifying the originals
    comps = comps_df.copy()
    props = properties_df.copy()
    
    # Step 1: Convert each comp to a training example
    # Label selected comps as positive examples (y=1)
    examples = []
    
    # For each order_id in comps_df
    order_ids = comps['order_id'].unique()
    logger.info(f"Processing {len(order_ids)} appraisal orders for training data")
    
    for order_id in order_ids:
        # Get comps for this order
        order_comps = comps[comps['order_id'] == order_id]
        
        # Find matching properties that are likely these comps
        for _, comp in order_comps.iterrows():
            # Create feature vector for this comp
            example = {
                'order_id': comp['order_id'],
                'structure_type': comp['structure_type'],
                'comp_rank': comp['comp_rank'],
                'distance_to_subject': comp['distance_to_subject'],
                'gla': comp['gla'],
                'age': comp['age'],
                'sale_price': comp['sale_price'],
                'bedrooms': comp['bedrooms'],
                'full_baths': comp['full_baths'],
                'half_baths': comp['half_baths'] if 'half_baths' in comp else 0,
                'is_selected': 1,  # This is a selected comp
                'address': comp['address'],
                'city': comp['city']
            }
            
            examples.append(example)
        
        # Add the top 3 properties (excluding the already selected comps) as negative examples
        order_props = extract_comp_candidates(
            {'order_id': order_id},
            props,
            exclude_addresses=[c['address'] for _, c in order_comps.iterrows()]
        )
        
        # Limit to top 3 negative examples per order
        if len(order_props) > 3:
            order_props = order_props.iloc[:3]
        
        for _, prop in order_props.iterrows():
            example = {
                'order_id': order_id,
                'structure_type': prop['structure_type'],
                'comp_rank': 99,  # Not selected
                'distance_to_subject': 99.0,  # Unknown
                'gla': prop['gla'],
                'age': prop['age'],
                'sale_price': prop['sale_price'],
                'bedrooms': prop['bedrooms'],
                'full_baths': prop['full_baths'] if 'full_baths' in prop else np.nan,
                'half_baths': prop['half_baths'] if 'half_baths' in prop else np.nan,
                'is_selected': 0,  # This is not a selected comp
                'address': prop['address'],
                'city': prop['city']
            }
            
            examples.append(example)
    
    # Convert examples to DataFrame
    train_df = pd.DataFrame(examples)
    
    # Fill missing values
    for col in ['half_baths', 'distance_to_subject']:
        if col in train_df.columns and train_df[col].isna().any():
            train_df[col] = train_df[col].fillna(0)
    
    for col in ['full_baths', 'bedrooms', 'age']:
        if col in train_df.columns and train_df[col].isna().any():
            train_df[col] = train_df[col].fillna(train_df[col].median())
    
    logger.info(f"Created {len(train_df)} training examples, {train_df['is_selected'].sum()} positive")
    
    # Ensure all feature columns are present
    for feature in FEATURES:
        if feature not in train_df.columns:
            if feature == 'lot_size':
                train_df[feature] = 0
            else:
                train_df[feature] = None
    
    # Extract features, target and groups
    X = train_df[FEATURES]
    y = train_df['is_selected']
    groups = train_df['order_id']
    
    logger.info(f"Final training data shape: {X.shape}")
    return X, y, groups

def extract_comp_candidates(
    subject_data: Dict, 
    properties_df: pd.DataFrame,
    features: List[str] = FEATURES,
    exclude_addresses: List[str] = None,
    use_order_id_filter: bool = True
) -> pd.DataFrame:
    """
    Extract candidate comparable properties for a subject property
    
    Args:
        subject_data: Dict with subject property data
        properties_df: DataFrame of all properties
        features: List of feature names to include
        exclude_addresses: List of addresses to exclude
        use_order_id_filter: Whether to filter by order_id (set to False for test properties)
        
    Returns:
        DataFrame of candidate properties with features
    """
    # Extract order ID and find related properties
    order_id = subject_data.get('order_id')
    
    if use_order_id_filter and order_id:
        # Filter by order_id as done in the notebook
        candidates = properties_df[properties_df['order_id'] == order_id].copy()
        logger.info(f"Extracted {len(candidates)} candidates for order_id {order_id}")
        
        if candidates.empty:
            logger.warning(f"No properties found with order_id {order_id}, using all properties instead")
            candidates = properties_df.copy()
    else:
        # Use all properties as candidates (fallback)
        candidates = properties_df.copy()
        
    # Ensure all required features are present (only the ones we need)
    for feature in features:
        if feature not in candidates.columns:
            candidates[feature] = np.nan if feature in NUMERIC_FEATS else ""
            
    # Filter out excluded addresses
    if exclude_addresses:
        candidates = candidates[~candidates['address'].str.lower().str.strip().isin(
            [addr.lower().strip() for addr in exclude_addresses]
        )]
    
    # Exclude the subject property itself by address
    subject_address = subject_data.get('address')
    if subject_address and 'address' in candidates.columns:
        candidates = candidates[
            candidates['address'].str.lower().str.strip() != subject_address.lower().strip()
        ]
    
    return candidates

def filter_by_structure_type(
    candidates_df: pd.DataFrame, 
    subject_structure_type: str, 
    include_similar: bool = True
) -> pd.DataFrame:
    """
    Filter candidates by structure type
    
    Args:
        candidates_df: DataFrame of candidate properties
        subject_structure_type: Structure type of the subject property
        include_similar: Whether to include similar structure types
        
    Returns:
        Filtered DataFrame
    """
    if include_similar:
        similar_types = get_similar_structure_types(subject_structure_type)
        return candidates_df[candidates_df['structure_type'].isin(similar_types)]
    else:
        return candidates_df[candidates_df['structure_type'] == subject_structure_type]

def get_similar_structure_types(structure_type: str) -> List[str]:
    """
    Get list of similar structure types for filtering
    
    Args:
        structure_type: Base structure type
        
    Returns:
        List of similar structure types
    """
    if not structure_type or pd.isna(structure_type):
        # If no structure type provided, return an empty list that will match nothing
        return []
        
    # Define groups of similar structure types (using more inclusive groupings)
    type_groups = {
        'detached': ['detached', 'single family', 'house', 'bungalow', 'sidesplit', 'backsplit', 
                    'single-family', '2-storey', '1 1/2 storey', '1 storey', 'link'],
        'semi-detached': ['semi-detached', 'semi', 'semi-d'],
        'townhouse': ['townhouse', 'row house', 'freehold townhouse', 'townhome', 'row', 'stacked'],
        'condo': ['condo', 'condominium', 'apartment', 'high rise', 'low rise', 'apartment-high-rise', 
                 'apartment-low-rise', 'high-rise', 'low-rise', 'strata'],
        'multi-family': ['duplex', 'triplex', 'multi-family', 'multi family', 'multiplex'],
        'rural': ['rural', 'country', 'farm', 'acreage', 'agricultural']
    }
    
    # Standardize the input type
    structure_type_lower = structure_type.lower()
    
    # Find which group the input type belongs to
    for group, types in type_groups.items():
        if any(t in structure_type_lower for t in types):
            logger.info(f"Structure type '{structure_type}' matched with group '{group}'")
            return types
    
    # If no match found, return the original type in a list
    logger.warning(f"No similar structure types found for '{structure_type}', using exact match only")
    return [structure_type]

def calculate_distance_features(
    subject_data: Dict, 
    candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate distance-based features between subject and candidates
    
    Args:
        subject_data: Dict with subject property data
        candidates_df: DataFrame of candidate properties
        
    Returns:
        DataFrame with added distance features
    """
    # Make a copy to avoid modifying the original
    df = candidates_df.copy()
    
    # Get subject coordinates
    subject_lat = subject_data.get('latitude')
    subject_lon = subject_data.get('longitude')
    
    if pd.isna(subject_lat) or pd.isna(subject_lon) or 'latitude' not in df.columns or 'longitude' not in df.columns:
        # Cannot calculate distance
        df['distance_km'] = np.nan
        return df
    
    # Calculate Haversine distance
    R = 6371  # Earth radius in km
    
    # Convert to radians
    lat1 = np.radians(subject_lat)
    lon1 = np.radians(subject_lon)
    lat2 = np.radians(df['latitude'])
    lon2 = np.radians(df['longitude'])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c
    
    df['distance_km'] = distances
    
    return df

def calculate_price_features(
    subject_data: Dict, 
    candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate price-related features
    
    Args:
        subject_data: Dict with subject property data
        candidates_df: DataFrame of candidate properties
        
    Returns:
        DataFrame with added price features
    """
    df = candidates_df.copy()
    
    # Calculate price per square foot
    if 'sale_price' in df.columns and 'gla' in df.columns:
        df['price_per_sqft'] = df['sale_price'] / df['gla']
    
    # Calculate age of sale (days since sale date)
    if 'sale_date' in df.columns:
        today = pd.Timestamp('today').normalize()
        df['days_since_sale'] = (today - pd.to_datetime(df['sale_date'])).dt.days
    
    return df

def calculate_similarity_score(
    subject_data: Dict, 
    candidates_df: pd.DataFrame,
    feature_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Calculate similarity score between subject and candidates
    
    Args:
        subject_data: Dict with subject property data
        candidates_df: DataFrame of candidate properties
        feature_weights: Dict mapping feature names to weights
        
    Returns:
        DataFrame with added similarity score
    """
    df = candidates_df.copy()
    
    if df.empty:
        return df
    
    # Default weights if not provided - updated to match our feature set
    if feature_weights is None:
        feature_weights = {
            'gla': 0.25,
            'age': 0.15,
            'bedrooms': 0.15,
            'full_baths': 0.15,
            'room_count': 0.10,
            'structure_type': 0.15,
            'stories': 0.05
        }
    
    # Normalize numeric features
    numeric_features = [f for f in feature_weights.keys() 
                         if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    for feature in numeric_features:
        if feature not in subject_data:
            continue
            
        subject_value = subject_data[feature]
        if pd.isna(subject_value):
            continue
            
        # Calculate absolute difference
        df[f'{feature}_diff'] = np.abs(df[feature] - subject_value)
        
        # Normalize to 0-1 range (1 = identical, 0 = most different)
        max_diff = df[f'{feature}_diff'].max()
        if max_diff > 0:
            df[f'{feature}_sim'] = 1 - (df[f'{feature}_diff'] / max_diff)
        else:
            df[f'{feature}_sim'] = 1.0
    
    # Calculate categorical feature similarity (1 if same, 0 if different)
    cat_features = [f for f in feature_weights.keys() 
                     if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]
    
    for feature in cat_features:
        if feature not in subject_data:
            continue
            
        subject_value = subject_data[feature]
        if pd.isna(subject_value):
            continue
            
        df[f'{feature}_sim'] = np.where(df[feature] == subject_value, 1.0, 0.0)
    
    # Calculate weighted similarity score
    similarity_features = [f'{f}_sim' for f in feature_weights.keys() 
                          if f'{f}_sim' in df.columns]
    
    if not similarity_features:
        df['similarity_score'] = 0.5  # Default if no features to compare
        return df
    
    # Calculate weighted sum
    df['similarity_score'] = 0
    total_weight = 0
    
    for feature in feature_weights.keys():
        sim_feature = f'{feature}_sim'
        if sim_feature in df.columns:
            weight = feature_weights[feature]
            df['similarity_score'] += df[sim_feature] * weight
            total_weight += weight
    
    # Normalize by total weight used
    if total_weight > 0:
        df['similarity_score'] /= total_weight
    
    return df 