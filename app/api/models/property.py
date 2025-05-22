from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class PropertyBase(BaseModel):
    """Base property model with common fields"""
    id: Optional[str] = None
    order_id: Optional[str] = None
    address: Optional[str] = None
    bedrooms: Optional[float] = None
    gla: Optional[float] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    structure_type: Optional[str] = None
    stories: Optional[str] = None
    room_count: Optional[float] = None
    full_baths: Optional[float] = None
    half_baths: Optional[float] = None
    lot_size: Optional[Union[str, float]] = None
    age: Optional[float] = None
    year_built: Optional[float] = None
    basement: Optional[str] = None
    heating: Optional[str] = None
    cooling: Optional[str] = None
    sale_date: Optional[str] = None
    sale_price: Optional[float] = None
    public_remarks: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PropertyCreate(PropertyBase):
    """Model for creating a new property"""
    order_id: str
    id: Optional[str] = None

class PropertyResponse(PropertyBase):
    """Property model for response with additional fields"""
    score: Optional[float] = None
    is_true_comp: Optional[int] = None

class SubjectProperty(PropertyBase):
    """Model for subject property"""
    order_id: str
    effective_date: Optional[datetime] = None

class PropertyRecommendationRequest(BaseModel):
    """Request model for property recommendations"""
    subject_property: PropertyBase
    max_recommendations: int = Field(default=3, ge=1, le=10)
    min_score_threshold: Optional[float] = Field(default=0.01, ge=0.0, le=1.0)
    count: Optional[int] = Field(default=3, ge=1, le=10)

class PropertyRecommendationResponse(BaseModel):
    """Response model for property recommendations"""
    subject_property: PropertyResponse
    recommendations: List[PropertyResponse]
    model_version: Optional[str] = None

class FeedbackBase(BaseModel):
    """Base feedback model"""
    subject_id: str
    subject_data: Dict[str, Any]
    recommended_properties: List[Dict[str, Any]]
    is_approved: bool
    selected_properties: List[Dict[str, Any]] = []
    comments: Optional[str] = None

class FeedbackSubmitRequest(FeedbackBase):
    """Request model for feedback submission"""
    pass

class FeedbackSubmitResponse(BaseModel):
    """Response model for feedback submission"""
    id: str
    status: str
    message: Optional[str] = None

class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics"""
    total_count: int
    positive_count: int
    negative_count: int
    approval_rate: float
    last_updated: Optional[str] = None
    last_retrain: Optional[str] = None

class RetrainRequest(BaseModel):
    """Request model for model retraining"""
    force: bool = False

class RetrainResponse(BaseModel):
    """Response model for model retraining"""
    status: str
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ExportResponse(BaseModel):
    """Response model for feedback export"""
    path: str
    count: int 