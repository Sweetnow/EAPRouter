"""
Data models for recommendation module
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class Item(BaseModel):
    """
    Item Model (movies, products, etc.)
    """
    
    item_id: int = Field(..., description="Item ID")
    name: str = Field(..., min_length=1, max_length=500, description="Item name")
    description: Optional[str] = Field(None, max_length=2000, description="Item description")
    category: Optional[str] = Field(None, max_length=100, description="Item category")
    created_at: datetime = Field(default_factory=datetime.now, description="Item creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        category_str = f" ({self.category})" if self.category else ""
        return f"Item {self.item_id}: {self.name}{category_str}"


class Rating(BaseModel):
    """
    User-Item Rating Model
    """
    
    user_id: int = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating value (1.0-5.0)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Rating timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Rating: User {self.user_id} -> Item {self.item_id} = {self.rating:.1f}"


class UserPreference(BaseModel):
    """
    User Preference Vector Model
    """
    
    user_id: int = Field(..., description="User ID")
    preference_vector: List[float] = Field(..., description="User preference embedding vector")
    learned_at: datetime = Field(default_factory=datetime.now, description="When the preference was learned")
    algorithm: str = Field(..., description="Algorithm used to learn the preference")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"UserPreference for User {self.user_id} (algorithm: {self.algorithm}, vector_dim: {len(self.preference_vector)})"


class FeedCache(BaseModel):
    """
    Feed Cache Model
    """
    
    user_id: int = Field(..., description="User ID")
    algorithm: str = Field(..., description="Algorithm used to generate the feed")
    items: List[int] = Field(..., description="List of recommended item IDs")
    generated_at: datetime = Field(default_factory=datetime.now, description="When the feed was generated")
    expires_at: datetime = Field(..., description="When the cache expires")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"FeedCache for User {self.user_id} (algorithm: {self.algorithm}, items: {len(self.items)}, expires: {self.expires_at})"


class RecommendationHistory(BaseModel):
    """
    Recommendation History Model
    """
    
    user_id: int = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID that was recommended")
    algorithm: str = Field(..., description="Algorithm used for recommendation")
    score: float = Field(..., description="Recommendation score")
    rank: int = Field(..., ge=1, description="Rank in the recommendation list")
    shown_at: datetime = Field(default_factory=datetime.now, description="When the recommendation was shown")
    clicked: Optional[bool] = Field(None, description="Whether the user clicked on the recommendation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        clicked_str = "Clicked" if self.clicked else "Not clicked" if self.clicked is False else "Unknown"
        return f"RecommendationHistory: User {self.user_id} -> Item {self.item_id} (rank: {self.rank}, score: {self.score:.3f}, {clicked_str})"

