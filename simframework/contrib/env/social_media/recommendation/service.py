"""
推荐服务层
"""

import asyncio
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from simframework.logger import get_logger
from .algorithms.core import RecommenderAlgorithm, RatingMatrix


@dataclass
class ServiceConfig:
    """
    推荐服务配置

    Args:
        cache_ttl: 缓存过期时间 (秒),默认 300秒
        max_batch_size: 最大批量推荐大小,默认 100
        timeout: 请求超时时间 (秒),默认 10秒
    """
    cache_ttl: int = 300
    max_batch_size: int = 100
    timeout: float = 10.0


class RecommendationService:
    """
    推荐服务
    """

    def __init__(
        self,
        algorithm: RecommenderAlgorithm,
        config: ServiceConfig = ServiceConfig()
    ):
        """
        初始化推荐服务

        Args:
            algorithm: 推荐算法实例
            config: 服务配置
        """
        self._algorithm = algorithm
        self._config = config

        # 缓存: cache_key -> (timestamp, result)
        self._cache: Dict[str, Tuple[float, List[Tuple[int, float]]]] = {}

        # 锁保护模型访问
        self._lock = asyncio.Lock()

        get_logger().info(
            f"RecommendationService 初始化: "
            f"algorithm={algorithm.get_algorithm_name()}, "
            f"cache_ttl={config.cache_ttl}s"
        )

    async def fit(self, data: RatingMatrix) -> None:
        """
        训练模型 (异步包装)

        Args:
            data: 评分矩阵
        """
        get_logger().info(f"开始训练模型: {data}")

        async with self._lock:
            # 在线程池中执行同步训练
            await asyncio.to_thread(self._algorithm.fit, data)

        # 训练后清空缓存
        self._cache.clear()

        get_logger().info("模型训练完成")

    async def predict(self, user_id: int, item_id: int) -> float:
        """
        预测评分 (异步包装)

        Args:
            user_id: 用户ID
            item_id: 物品ID

        Returns:
            预测评分 (1.0-5.0)
        """
        return await asyncio.to_thread(
            self._algorithm.predict,
            user_id,
            item_id
        )

    async def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_rated: bool = True,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        生成推荐 (带缓存)

        Args:
            user_id: 用户ID
            n: 推荐数量
            exclude_rated: 是否排除已评分物品 (暂未实现,保留接口)
            exclude_ids: 额外要排除的物品ID集合

        Returns:
            [(item_id, score), ...] 按 score 降序排列
        """
        # 检查缓存
        exclude_set = exclude_ids or set()
        cache_key = f"{user_id}:{n}:{len(exclude_set)}"

        if cache_key in self._cache:
            timestamp, cached_result = self._cache[cache_key]
            if time.time() - timestamp < self._config.cache_ttl:
                get_logger().debug(f"缓存命中: user={user_id}")
                return cached_result

        # 调用算法生成推荐
        result = await asyncio.to_thread(
            self._algorithm.recommend,
            user_id,
            n,
            exclude_set
        )

        # 更新缓存
        self._cache[cache_key] = (time.time(), result)

        get_logger().debug(
            f"为用户 {user_id} 生成 {len(result)} 条推荐"
        )

        return result

    async def batch_recommend(
        self,
        user_ids: List[int],
        n: int = 20,
        exclude_ids: Optional[Dict[int, Set[int]]] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量生成推荐

        Args:
            user_ids: 用户ID列表
            n: 每个用户的推荐数量
            exclude_ids: 每个用户要排除的物品ID字典

        Returns:
            {user_id: [(item_id, score), ...], ...}
        """
        exclude_dict = exclude_ids or {}

        # 并发执行推荐
        tasks = [
            self.recommend(
                user_id=uid,
                n=n,
                exclude_ids=exclude_dict.get(uid)
            )
            for uid in user_ids
        ]

        results = await asyncio.gather(*tasks)

        return {
            user_id: result
            for user_id, result in zip(user_ids, results)
        }

    async def save_model(self, path: str) -> None:
        """
        保存模型 (异步包装)

        Args:
            path: 模型保存路径
        """
        async with self._lock:
            await asyncio.to_thread(self._algorithm.save, path)

        get_logger().info(f"模型已保存到 {path}")

    async def load_model(self, path: str) -> None:
        """
        加载模型 (异步包装)

        Args:
            path: 模型文件路径
        """
        async with self._lock:
            await asyncio.to_thread(self._algorithm.load, path)

        # 加载后清空缓存
        self._cache.clear()

        get_logger().info(f"模型已从 {path} 加载")

    def get_algorithm_info(self) -> Dict[str, any]:
        """
        获取算法信息

        Returns:
            算法信息字典
        """
        return self._algorithm.get_algorithm_info()

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        get_logger().info("推荐缓存已清空")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        获取缓存统计

        Returns:
            缓存统计信息
        """
        current_time = time.time()
        valid_entries = sum(
            1 for timestamp, _ in self._cache.values()
            if current_time - timestamp < self._config.cache_ttl
        )

        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'cache_ttl': self._config.cache_ttl
        }
