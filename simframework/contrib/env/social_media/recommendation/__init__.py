"""
Recommendation Module for SocialMediaSpace

- RecommenderAlgorithm: 统一的算法接口
- RatingMatrix: 统一的数据格式
- MFRecommender: MF算法实现
- RecommendationService: 核心推荐服务
- IncrementalTrainer: 增量训练器 (可选)
"""

from .models import Item, Rating, UserPreference, FeedCache, RecommendationHistory
from .storage import RecommendationStorageManager

from .algorithms.core import RecommenderAlgorithm, RatingMatrix
from .algorithms.mf import MFRecommender, MFConfig
from .service import RecommendationService, ServiceConfig
from .trainer import IncrementalTrainer, TrainerConfig

__all__ = [
    # 数据模型
    "Item",
    "Rating",
    "UserPreference",
    "FeedCache",
    "RecommendationHistory",
    "RecommendationStorageManager",

    "RecommenderAlgorithm",
    "RatingMatrix",
    "MFRecommender",
    "MFConfig",
    "RecommendationService",
    "ServiceConfig",
    "IncrementalTrainer",
    "TrainerConfig",
]
