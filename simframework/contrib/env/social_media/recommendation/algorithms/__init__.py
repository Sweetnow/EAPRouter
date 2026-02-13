"""
Recommendation Algorithms

统一算法接口和实现:
- RecommenderAlgorithm: 统一算法抽象基类
- RatingMatrix: 统一评分数据格式
- MFRecommender: MF (矩阵分解) 算法实现
- SASRecRecommender: SASRec (序列推荐) 算法实现
- NCFRecommender: NCF (神经协同过滤) 算法实现
- DeepFMRecommender: DeepFM (深度因子分解机) 算法实现
- DINRecommender: DIN (深度兴趣网络) 算法实现
- LightGCNRecommender: LightGCN (轻量级图卷积网络) 算法实现
"""

from .core import RecommenderAlgorithm, RatingMatrix
from .mf import MFRecommender, MFConfig
from .sasrec import SASRecRecommender, SASRecConfig
from .ncf import NCFRecommender, NCFConfig
from .deepfm import DeepFMRecommender, DeepFMConfig
from .din import DINRecommender, DINConfig
from .lightgcn import LightGCNRecommender, LightGCNConfig

__all__ = [
    "RecommenderAlgorithm",
    "RatingMatrix",
    "MFRecommender",
    "MFConfig",
    "SASRecRecommender",
    "SASRecConfig",
    "NCFRecommender",
    "NCFConfig",
    "DeepFMRecommender",
    "DeepFMConfig",
    "DINRecommender",
    "DINConfig",
    "LightGCNRecommender",
    "LightGCNConfig",
]
