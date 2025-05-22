from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class FeedbackBase(BaseModel):
    """Base model for feedback data"""
    subject_id: str
    is_approved: bool
    comments: Optional[str] = None

class FeedbackCreate(FeedbackBase):
    """Model for creating feedback"""
    recommended_property_ids: List[str]
    selected_property_ids: Optional[List[str]] = None

class FeedbackCreateDetailed(FeedbackBase):
    """Model for creating detailed feedback with property data"""
    recommended_properties: List[Dict[str, Any]]
    selected_properties: Optional[List[Dict[str, Any]]] = None
    subject_data: Dict[str, Any]

class FeedbackResponse(FeedbackBase):
    """Model for feedback response"""
    id: str
    timestamp: datetime
    processed: bool

    class Config:
        orm_mode = True

class FeedbackSubmitResponse(BaseModel):
    """Model for feedback submission response"""
    feedback_id: str
    status: str
    message: Optional[str] = None
    new_feedback_count: int
    retrain_threshold: int

class FeedbackStats(BaseModel):
    """Model for feedback statistics"""
    total_count: int
    positive_count: int
    negative_count: int
    processed_count: int
    unprocessed_count: int
    approval_rate: float
    last_updated: Optional[datetime] = None
    last_retrain: Optional[datetime] = None

class ModelRetrainRequest(BaseModel):
    """Model for model retraining request"""
    force: bool = Field(False, description="Force retraining even if threshold not reached")

class ModelRetrainResponse(BaseModel):
    """Model for model retraining response"""
    status: str
    reason: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    processed_feedback_count: Optional[int] = None 