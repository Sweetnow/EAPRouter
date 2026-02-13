# -*- coding: utf-8 -*-

"""
列出可用的模型列表
"""

import os
import json
import requests
from typing import List, Optional, Dict, Any

__all__ = ["get_available_models"]


def get_available_models(
    api_base: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 20
) -> List[Dict[str, str]]:
    """
    获取当前可用的模型列表，输出 JSON 结构，仅包含关键的 'id' 字段。

    Args:
        api_base: API 基础 URL，若未提供则使用默认值 https://cloud.infini-ai.com
        api_key: API 密钥，若未提供则从环境变量读取（需要调用者提前加载 .env 文件）
        timeout: 请求超时时间（秒）

    Returns:
        List[Dict[str, str]]: 模型列表，格式如下：
        [
            {"id": "qwen2.5-72b-instruct"},
            {"id": "deepseek-r1"}
        ]

    Raises:
        ValueError: 当缺少必要的 API_KEY 时
        RuntimeError: 当网络请求失败或响应格式异常时
    """
    base = api_base or "https://cloud.infini-ai.com"
    key = api_key or os.getenv("API_KEY", "").strip()

    if not key:
        raise ValueError("缺少 API_KEY，请通过入参或 .env 提供")

    url = f"{base}/maas/v1/models"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"网络请求异常：{e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"请求失败：HTTP {resp.status_code} - {resp.text}")

    try:
        body = resp.json()
    except ValueError as e:
        raise RuntimeError(f"响应非 JSON：{resp.text}") from e

    if (
        not isinstance(body, dict)
        or body.get("object") != "list"
        or not isinstance(body.get("data"), list)
    ):
        raise RuntimeError(f"返回结构异常：{body}")

    models: List[Dict[str, str]] = []
    for item in body["data"]:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append({"id": item["id"]})

    return models


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 入口脚本负责加载环境变量
    load_dotenv()
    
    try:
        # 获取模型列表
        models_data = get_available_models()

        # 输出 JSON 格式
        print(json.dumps(models_data, indent=2, ensure_ascii=False))
    except Exception as e:
        # 发生错误
        error_response = {"error": str(e)}
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
