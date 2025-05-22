from fastapi import APIRouter, Depends, HTTPException, status
import logging
import os
from typing import Dict, Any

from app.api.models.property import (
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    FeedbackStatsResponse,
    RetrainRequest,
    RetrainResponse,
    ExportResponse
)
from app.ml.feedback_learning import FeedbackLearningSystem
from app.ml.model import PropertyRecommendationModel
from app.api.routes.properties import get_model
from app.ml import initialize_feedback_system

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/feedback", tags=["feedback"])

# Feedback system instance
FEEDBACK_SYSTEM = None
FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "data/feedback/feedback_database.json")
TRAINING_DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "data/processed/training_data.pkl")
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "data/models/recommendation_model.pkl")
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "10"))

def get_feedback_system(model: PropertyRecommendationModel = Depends(get_model)):
    """
    Get or create the feedback learning system
    
    Returns:
        FeedbackLearningSystem instance
    """
    global FEEDBACK_SYSTEM
    
    if FEEDBACK_SYSTEM is None:
        try:
            FEEDBACK_SYSTEM = initialize_feedback_system(
                model=model,
                feedback_db_path=FEEDBACK_DB_PATH,
                training_data_path=TRAINING_DATA_PATH,
                model_save_path=MODEL_SAVE_PATH,
                retrain_threshold=RETRAIN_THRESHOLD
            )
            logger.info(f"Feedback learning system initialized with database at {FEEDBACK_DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing feedback system: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize feedback system: {str(e)}"
            )
    
    return FEEDBACK_SYSTEM

@router.post("/submit", response_model=FeedbackSubmitResponse)
async def submit_feedback(
    feedback: FeedbackSubmitRequest,
    feedback_system: FeedbackLearningSystem = Depends(get_feedback_system)
):
    """
    Submit feedback for property recommendations
    """
    try:
        # Add feedback to the system
        result = feedback_system.add_feedback(
            subject_id=feedback.subject_id,
            subject_data=feedback.subject_data,
            recommended_properties=feedback.recommended_properties,
            is_approved=feedback.is_approved,
            selected_properties=feedback.selected_properties,
            comments=feedback.comments
        )
        
        # Return the result
        return FeedbackSubmitResponse(
            id=result.get("feedback_id", "unknown"),
            status="success",
            message="Feedback submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_statistics(
    feedback_system: FeedbackLearningSystem = Depends(get_feedback_system)
):
    """
    Get feedback statistics
    """
    try:
        # Get statistics from the feedback system
        stats = feedback_system.get_feedback_stats()
        return FeedbackStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error getting feedback statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback statistics: {str(e)}"
        )

@router.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainRequest,
    feedback_system: FeedbackLearningSystem = Depends(get_feedback_system)
):
    """
    Retrain the model with feedback data
    """
    try:
        # Retrain the model
        result = feedback_system.retrain_model(force=request.force)
        
        if result.get("retrained", False):
            return RetrainResponse(
                status="success",
                reason="Model retrained successfully",
                details={
                    "metrics": result.get("metrics"),
                    "processed_feedback_count": result.get("processed_feedback_count", 0)
                }
            )
        else:
            return RetrainResponse(
                status="skipped",
                reason=result.get("reason", "Not enough feedback to retrain"),
                details={
                    "processed_feedback_count": result.get("processed_feedback_count", 0)
                }
            )
    
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrain model: {str(e)}"
        )

@router.get("/export", response_model=ExportResponse)
async def export_feedback(
    feedback_system: FeedbackLearningSystem = Depends(get_feedback_system)
):
    """
    Export feedback data to CSV
    """
    try:
        # Export feedback data
        export_path = "data/feedback/exported_feedback.csv"
        result_path = feedback_system.export_feedback_to_csv(export_path)
        
        return ExportResponse(
            path=result_path,
            count=feedback_system.get_feedback_count()
        )
    
    except Exception as e:
        logger.error(f"Error exporting feedback data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export feedback data: {str(e)}"
        ) 