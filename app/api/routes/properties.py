from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
import os
import numpy as np

from app.api.models.property import (
    PropertyRecommendationRequest,
    PropertyRecommendationResponse,
    PropertyResponse
)
from app.ml.model import PropertyRecommendationModel
from app.ml.utils import format_property_record, log_prediction
from app.ml.feature_engineering import FEATURES, extract_comp_candidates, filter_by_structure_type, calculate_distance_features, calculate_similarity_score
from app.ml import initialize_model_from_data
from app.ml.data_processing import load_and_process_data, deduplicate_properties

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/properties", tags=["properties"])

# Create model instance - will be loaded later
MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/recommendation_model.pkl")
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "appraisals_dataset.json")

# Store processed properties for recommendations
PROPERTIES_DF = None

def get_model():
    """
    Get or load the recommendation model
    
    Returns:
        PropertyRecommendationModel instance
    """
    global MODEL, PROPERTIES_DF
    
    if MODEL is None:
        try:
            # Initialize model from data
            MODEL = initialize_model_from_data(
                json_path=JSON_DATA_PATH,
                model_path=MODEL_PATH
            )
            logger.info(f"Model initialized successfully")
            
            # Load and process property data for recommendations
            if PROPERTIES_DF is None:
                logger.info(f"Loading property data from {JSON_DATA_PATH}")
                subjects_df, comps_df, properties_df = load_and_process_data(JSON_DATA_PATH)
                PROPERTIES_DF = deduplicate_properties(properties_df)
                
                # Ensure property IDs are strings for consistency with model's test_property_ids
                if 'id' in PROPERTIES_DF.columns:
                    PROPERTIES_DF['id'] = PROPERTIES_DF['id'].astype(str)
                    logger.info(f"Converted property IDs to strings: {PROPERTIES_DF['id'].dtype}")
                
                logger.info(f"Loaded {len(PROPERTIES_DF)} properties for recommendations")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load recommendation model: {str(e)}"
            )
    
    return MODEL

def get_properties_df():
    """
    Get the properties DataFrame
    
    Returns:
        DataFrame of properties
    """
    global PROPERTIES_DF
    
    if PROPERTIES_DF is None:
        # Ensure model and data are loaded
        get_model()
    
    return PROPERTIES_DF

@router.post("/retrain-model")
async def retrain_model():
    """
    Force retrain the recommendation model
    """
    global MODEL, PROPERTIES_DF
    
    try:
        logger.info("Forcing model retraining")
        
        # Force retrain the model
        MODEL = initialize_model_from_data(
            json_path=JSON_DATA_PATH,
            model_path=MODEL_PATH,
            force_retrain=True
        )
        
        # Reload the properties DataFrame
        logger.info(f"Reloading property data from {JSON_DATA_PATH}")
        subjects_df, comps_df, properties_df = load_and_process_data(JSON_DATA_PATH)
        PROPERTIES_DF = deduplicate_properties(properties_df)
        
        # Ensure property IDs are strings for consistency with model's test_property_ids
        if 'id' in PROPERTIES_DF.columns:
            PROPERTIES_DF['id'] = PROPERTIES_DF['id'].astype(str)
            logger.info(f"Converted property IDs to strings: {PROPERTIES_DF['id'].dtype}")
            
            # Check if model's test_property_ids are strings
            test_id_sample = list(MODEL.test_property_ids)[:5] if MODEL.test_property_ids else []
            logger.info(f"Model test ID samples: {test_id_sample}")
            logger.info(f"Properties DF ID samples: {PROPERTIES_DF['id'].head(5).tolist()}")
        
        # Count matching test properties
        matching_ids = PROPERTIES_DF['id'].isin(MODEL.test_property_ids)
        matches_count = matching_ids.sum()
        logger.info(f"Found {matches_count} properties that match test_property_ids after retraining")
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "test_properties_count": len(MODEL.test_property_ids),
            "matching_properties_count": matches_count
        }
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        logger.exception("Full exception details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrain model: {str(e)}"
        )

@router.post("/recommend", response_model=PropertyRecommendationResponse)
async def get_property_recommendations(
    request: PropertyRecommendationRequest,
    model: PropertyRecommendationModel = Depends(get_model),
    properties_df: pd.DataFrame = Depends(get_properties_df)
):
    """
    Get property recommendations for a subject property following notebook approach
    """
    try:
        # Convert subject property to dictionary
        subject_dict = request.subject_property.dict()
        subject_address = subject_dict.get('address', '')
        order_id = subject_dict.get('order_id', '')
        
        logger.info(f"Processing recommendation request for subject {order_id}: {subject_address}")
        
        # Log subject property details
        logger.info(f"Subject property details:")
        for feat in FEATURES:
            if feat in subject_dict:
                logger.info(f"  {feat}: {subject_dict.get(feat)}")
        
        # Get properties for this subject's order_id - this follows the notebook approach
        # These are the candidates that were available for this subject
        order_properties = properties_df[properties_df['order_id'] == order_id].copy()
        
        # Exclude the subject property itself
        if 'address' in order_properties.columns and subject_address:
            order_properties = order_properties[order_properties['address'].str.lower() != subject_address.lower()]
        
        logger.info(f"Found {len(order_properties)} candidates for order_id {order_id}")
        
        if order_properties.empty:
            logger.warning(f"No candidates found for order_id {order_id}. Using all properties as fallback.")
            # As a fallback, use properties with similar features
            order_properties = properties_df.copy()
        
        # Load comp data to identify true comps
        _, comps_df, _ = load_and_process_data(JSON_DATA_PATH)
        
        # Filter comps for this order
        order_comps = comps_df[comps_df['order_id'] == order_id].copy()
        
        if not order_comps.empty:
            # Join comp_rank information to our properties
            order_properties = order_properties.merge(
                order_comps[['address', 'comp_rank', 'distance_to_subject']],
                on='address',
                how='left'
            )
            
            logger.info(f"Found {len(order_comps)} comps for order_id {order_id}")
            logger.info(f"True comp addresses: {order_comps['address'].tolist()}")
        
        # Create mappings for true comp injection
        comp_mapping = {
            'sale_price': 'sale_price',
            'sale_date': 'sale_date',
            'age': 'age',
            'gla': 'gla',
            'structure_type': 'structure_type',
            'stories': 'stories',
            'bedrooms': 'bedrooms',
            'bed_count': 'bedrooms',
            'full_baths': 'full_baths',
            'bath_count': 'full_baths',
            'half_baths': 'half_baths',
            'room_count': 'room_count',
            'address': 'address',
            'city': 'city',
            'comp_rank': 'comp_rank'
        }
        
        # Prepare true comps dataframe for injection
        true_comps_for_injection = None
        if not order_comps.empty:
            # Rename columns to match property schema
            true_comps_for_injection = order_comps.rename(columns=comp_mapping)
            
            # Set is_true_comp flag
            true_comps_for_injection['is_true_comp'] = 1
            
            # Set ID and order_id for true comps to prevent validation errors
            for i, row in true_comps_for_injection.iterrows():
                if pd.isna(row.get('id')):
                    # Generate a unique ID for this true comp
                    true_comps_for_injection.at[i, 'id'] = f"{order_id}_true_comp_{i}"
                
                # Ensure order_id is set
                true_comps_for_injection.at[i, 'order_id'] = order_id
                
                # Ensure other required fields are not NaN
                for field in ['city', 'province']:
                    if pd.isna(row.get(field)):
                        true_comps_for_injection.at[i, field] = ""
            
            logger.info(f"Prepared {len(true_comps_for_injection)} true comps for possible injection")
        
        # Generate recommendations using model - passing subject_dict and true_comps for injection
        recommendations = model.recommend_comps(
            order_properties,
            subject_data=subject_dict,
            n_recommendations=request.max_recommendations,
            threshold=request.min_score_threshold,
            true_comps_df=true_comps_for_injection
        )
        
        # Log the top recommendations
        if not recommendations.empty:
            logger.info(f"Top recommendations for subject {order_id}:")
            for i, (_, prop) in enumerate(recommendations.iterrows(), 1):
                prop_addr = prop.get('address', 'Unknown')
                prop_score = prop.get('score', 0)
                is_true_comp = prop.get('is_true_comp', 0)
                
                comp_status = "✓ TRUE COMP" if is_true_comp else ""
                logger.info(f"  {i}. {prop_addr} (score: {prop_score:.3f}) {comp_status}")
        
        # Format response
        formatted_recommendations = []
        for _, prop in recommendations.iterrows():
            prop_dict = {col: prop.get(col) for col in prop.index}
            formatted_prop = format_property_record(prop_dict)
            
            # Add score and is_true_comp from the model predictions
            formatted_prop['score'] = prop_dict.get('score', 0)
            formatted_prop['is_true_comp'] = prop_dict.get('is_true_comp', 0)
            
            formatted_recommendations.append(PropertyResponse(**formatted_prop))
        
        # Format subject property - IMPORTANT: remove sale_price from subject
        subject_formatted = format_property_record(subject_dict)
        subject_formatted.pop('sale_price', None)  # Remove sale_price from subject
        
        # Log the prediction
        log_prediction(
            subject_id=subject_dict.get('order_id', 'unknown'),
            subject_data=subject_dict,
            predictions=[prop.dict() for prop in formatted_recommendations]
        )
        
        # Return response
        return PropertyRecommendationResponse(
            subject_property=PropertyResponse(**subject_formatted),
            recommendations=formatted_recommendations,
            model_version=getattr(model, 'version', '1.0.0')
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.exception("Full exception details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@router.get("/test-subjects", response_model=List[PropertyResponse])
async def get_test_subjects(
    limit: int = 20,
    properties_df: pd.DataFrame = Depends(get_properties_df)
):
    """
    Get a list of test subjects from the notebook's test split
    """
    try:
        # Get model to access test property IDs
        model = get_model()
        
        if not hasattr(model, 'test_property_ids') or not model.test_property_ids:
            logger.warning("No test property IDs found in model")
            return []
        
        # Load subject data from appraisals file
        from app.ml.data_processing import load_and_process_data
        subjects_df, comps_df, _ = load_and_process_data(JSON_DATA_PATH)
        
        # Find unique test subject order_ids
        test_subjects = []
        
        # Find unique test subject order_ids - these are the 18 test subjects from the notebook
        for order_id in subjects_df['order_id'].unique():
            # Check if any property in this order is in the test set
            order_props = properties_df[properties_df['order_id'] == order_id]
            if any(str(prop_id) in model.test_property_ids for prop_id in order_props['id'].astype(str)):
                test_subjects.append(order_id)
        
        logger.info(f"Found {len(test_subjects)} unique test subject order_ids")
        
        if not test_subjects:
            logger.warning("No test subjects found. Returning sample of subjects.")
            # As fallback, use some random subjects
            test_subjects = properties_df['order_id'].unique()[:min(limit, len(properties_df['order_id'].unique()))]
        
        # Get one subject property per order_id
        subjects = []
        for order_id in test_subjects[:limit]:  # Limit to requested number
            # Get the subject property for this order_id
            subject_row = subjects_df[subjects_df['order_id'] == order_id]
            
            if not subject_row.empty:
                # Format the subject property
                prop_dict = subject_row.iloc[0].to_dict()
                formatted_prop = format_property_record(prop_dict)
                
                # IMPORTANT: Remove sale_price from subject
                formatted_prop.pop('sale_price', None)
                
                # CRITICAL: Ensure id field is set correctly for frontend to find the property
                formatted_prop['id'] = order_id
                
                subjects.append(PropertyResponse(**formatted_prop))
        
        logger.info(f"Returning {len(subjects)} test subjects")
        return subjects
    
    except Exception as e:
        logger.error(f"Error getting test subjects: {str(e)}")
        logger.exception("Full exception details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get test subjects: {str(e)}"
        ) 