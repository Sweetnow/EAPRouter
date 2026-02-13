import asyncio
import json
import logging
import os
import pickle
import shutil
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# load_dotenv(".env.openrouter")
load_dotenv()

from simframework.contrib.env.mobility_space import MobilitySpace
from simframework.contrib.env.event_space import EventSpace
from simframework.contrib.env.simple_social_space import SimpleSocialSpace
from simframework.agent import PersonAgent
from simframework.env import CodeGenRouter
from simframework.society import SimFramework
from simframework.logger import setup_logging, get_logger


def _calculate_gyration_radius(trajectories: list) -> float:
    """
    计算回旋半径（Radius of Gyration）
    
    回旋半径是从轨迹质心到各个位置点的平均距离的均方根。
    
    Args:
        trajectories: 轨迹列表，每个元素是 (x, y) 坐标对
    
    Returns:
        回旋半径（单位：米）
    """
    if len(trajectories) == 0:
        return 0.0
    
    trajectories = np.array(trajectories)
    # 计算轨迹的质心
    centroid = trajectories.mean(axis=0)
    
    # 计算每个点到质心的距离
    distances = np.linalg.norm(trajectories - centroid, axis=1)
    
    # 计算均方根距离（回旋半径）
    gyration_radius = np.sqrt(np.mean(distances ** 2))
    
    return float(gyration_radius)


async def main(
    logger,
    num_agents: int = 40,
    profile_start_idx: int = 60,
):
    """
    运行 DailyMobility Benchmark

    实验设置：
    - 模拟起点：当日早上 00:00:00 (UTC)
    - 时间步长：15 分钟 = 900 秒
    - 总步数：97 步（覆盖 24+ 小时）
    
    数据统计：
    - 轨迹数据：每个agent的移动轨迹（(x, y) 坐标列表）
    - 访问的AOI：每个agent访问过的AOI集合
    - 回旋半径：衡量agent活动范围的指标
    - 日均访问地点数：每个agent访问的唯一地点数
    """
    logger.info("\n" + "=" * 80)
    logger.info("【DailyMobility Benchmark】")
    logger.info("=" * 80)
    logger.info("实验设置：")
    logger.info("  - 起始时间: 当日早上 00:00:00 (UTC)")
    logger.info("  - 时间步长: 15 分钟 (900 秒)")
    logger.info("  - 总步数: 97 步")
    logger.info(f"  - Agent 数量: {num_agents}")
    logger.info("=" * 80)

    # 实验参数
    # 从早上 7 点开始模拟
    START_TIME = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    TIME_STEP_MINUTES = 15  # 15 分钟
    TIME_STEP_SECONDS = TIME_STEP_MINUTES * 60  # 900 秒
    TOTAL_STEPS = 97

    # 用于存储需要清理的环境
    mobility_env = None
    event_space = None
    env_router = None
    agents = []

    # ==================== 加载 Profiles ====================
    logger.info("\n【步骤1/4】加载 profiles.json...")

    profiles_path = os.path.join(os.path.dirname(__file__), "profiles.json")
    if not os.path.exists(profiles_path):
        logger.error(f"  ❌ profiles.json 文件不存在: {profiles_path}")
        return

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    logger.info(f"  ✓ 加载了 {len(profiles)} 个 agent profiles")

    # 限制 agent 数量
    if num_agents > len(profiles):
        logger.warning(
            f"  ⚠ 请求的 agent 数量 ({num_agents}) 超过 profiles 数量 ({len(profiles)})，使用全部 {len(profiles)} 个"
        )
        num_agents = len(profiles)

    profiles_to_use = profiles[profile_start_idx : profile_start_idx + num_agents]

    # 【关键修复】动态获取实际的 agent_ids，而不是硬编码 1-num_agents
    actual_agent_ids = [p["id"] for p in profiles_to_use]
    logger.info(f"  ✓ 实际 Agent IDs: {actual_agent_ids}")

    # ==================== 初始化环境 ====================
    logger.info("\n【步骤2/4】初始化环境...")

    import tempfile
    # 使用 tempfile 创建临时目录
    chroma_base_dir = tempfile.mkdtemp(prefix="chroma_memories_")
    logger.info(f"  ✓ 创建临时chroma目录: {chroma_base_dir}")

    # ==================== 创建 Agents ====================
    logger.info(f"\n【步骤3/4】创建 {num_agents} 个 Agents...")

    agent_args = []
    mobility_persons = []
    date_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    for profile in profiles_to_use:
        agent_id = profile["id"]

        # 为每个 agent 创建独立的 chroma 路径
        agent_chroma_path = os.path.join(
            chroma_base_dir, f"agent_{agent_id}_{date_time_str}"
        )
        os.makedirs(agent_chroma_path, exist_ok=True)
        agent_sqlite_path = os.path.join(chroma_base_dir, f"agent_{agent_id}.db")
        # 只确保父目录存在（chroma_base_dir 已存在，其实可省略）
        os.makedirs(os.path.dirname(agent_sqlite_path), exist_ok=True)

        # 创建 Agent 特定的 memory 配置
        agent_memory_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"agent_{agent_id}_memories",
                    "path": agent_chroma_path,
                },
            },
            "storage_config": {
                "provider": "sqlite",
                "path": agent_sqlite_path,
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen2.5-14b-instruct",
                    "api_key": os.getenv("INFINI_API_KEY"),
                    "openai_base_url": "https://cloud.infini-ai.com/maas/v1",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "bge-m3",
                    "api_key": os.getenv("INFINI_API_KEY"),
                    "openai_base_url": "https://cloud.infini-ai.com/maas/v1",
                    "embedding_dims": 1024,
                },
            },
        }

        # 创建 agent（使用 profile 中的详细信息）
        # 构建个人资料字符串
        profile_text = f"My name is Agent-{agent_id}, age {profile.get('age', 30)}, gender {profile.get('gender', 'Unknown')}, education {profile.get('education', 'Unknown')}, occupation {profile.get('occupation', 'Unknown')}, home at {profile['home']}, work at {profile['work']}"

        agent_args.append(
            {
                "id": agent_id,
                "profile": profile_text,
                "memory_config": agent_memory_config,
                "world_description": "",
            }
        )
        mobility_persons.append(
            {
                "id": agent_id,
                "position": {
                    "kind": "aoi",
                    "aoi_id": profile["home"],
                },
            }
        )

    # 创建 MobilitySpace 环境
    # 使用相对路径而不是硬编码的 /root 路径
    home_dir = os.path.join(os.path.expanduser("~"), "simframework_data")
    map_path = os.path.join(home_dir, "beijing.pb")
    os.makedirs(home_dir, exist_ok=True)

    mobility_env = MobilitySpace(map_path, home_dir, persons=mobility_persons)
    # person = await mobility_env.get_person(1)
    # print(person)
    # input("Press Enter to continue...")
    event_space = EventSpace()

    # 创建 CodeGenRouter
    env_router = CodeGenRouter(
        env_modules=[mobility_env, event_space],
        log_path=f"logs/instruction_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl",
    )

    # 保存 pyi 代码
    with open("tools_pyi.pyi", "w") as f:
        f.write(env_router._tools_pyi_dict[(False, None)])

    # 生成世界描述
    world_description = await env_router.generate_world_description_from_tools()

    print("--------------------------------")
    print(world_description)
    print("--------------------------------")

    # 更新 agent_args 中的 world_description
    for args in agent_args:
        args["world_description"] = world_description

    # 实际初始化agents
    agents = [PersonAgent(**args) for args in agent_args]

    society = SimFramework(
        agents=agents,
        env_router=env_router,
        start_t=START_TIME,
    )
    await society.init()

    await society.run(num_steps=TOTAL_STEPS, tick=TIME_STEP_SECONDS)

    # ==================== 提取移动相关数据 ====================
    logger.info(f"\n【步骤5/5】提取移动统计数据...")
    
    # 从 MobilitySpace 环境中获取移动相关数据
    trajectories_dict = mobility_env.get_all_persons_trajectories()
    visited_aois_dict = mobility_env.get_all_persons_visited_aois()
    
    # 计算各项指标
    gyration_radius_list = []
    daily_location_numbers_list = []
    trajectory_lengths = []
    
    for agent_id in actual_agent_ids:
        # 获取该agent的轨迹
        trajectory = trajectories_dict.get(agent_id, [])
        visited_aois = visited_aois_dict.get(agent_id, set())
        
        # 计算回旋半径
        gr = _calculate_gyration_radius(trajectory)
        gyration_radius_list.append(gr)
        
        # 计算访问的唯一AOI数量
        dln = len(visited_aois)
        daily_location_numbers_list.append(dln)
        
        # 记录轨迹长度
        trajectory_lengths.append(len(trajectory))
        
        logger.info(f"  Agent {agent_id}:")
        logger.info(f"    - 轨迹点数: {len(trajectory)}")
        logger.info(f"    - 访问AOI数: {dln}")
        logger.info(f"    - 回旋半径: {gr:.2f} 米")
        if len(visited_aois) > 0:
            logger.info(f"    - 访问的AOI ID: {sorted(visited_aois)[:5]}{'...' if len(visited_aois) > 5 else ''}")
    
    # 转换为 numpy 数组
    results = {
        "gyration_radius": np.array(gyration_radius_list, dtype=np.float64),
        "daily_location_numbers": np.array(daily_location_numbers_list, dtype=np.int32),
        "trajectories": trajectories_dict,  # 保留原始轨迹数据
        "visited_aois": visited_aois_dict,  # 保留访问的AOI数据
    }
    
    logger.info(f"\n  ✓ 数据提取完成")
    logger.info(f"    - gyration_radius shape: {results['gyration_radius'].shape}")
    logger.info(f"    - gyration_radius mean: {results['gyration_radius'].mean():.2f} 米")
    logger.info(f"    - gyration_radius std: {results['gyration_radius'].std():.2f} 米")
    logger.info(f"    - daily_location_numbers shape: {results['daily_location_numbers'].shape}")
    logger.info(f"    - daily_location_numbers mean: {results['daily_location_numbers'].mean():.2f}")
    logger.info(f"    - daily_location_numbers max: {results['daily_location_numbers'].max()}")
    
    # ==================== 保存结果 ====================
    logger.info(f"\n【保存结果】")
    
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"daily_mobility_results_{timestamp}.pkl")
    
    # 准备保存的数据
    save_data = {
        "results": {
            "gyration_radius": results["gyration_radius"],
            "daily_location_numbers": results["daily_location_numbers"],
        },
        "trajectories": results["trajectories"],
        "visited_aois": results["visited_aois"],
        "metadata": {
            "num_agents": num_agents,
            "actual_agent_ids": actual_agent_ids,
            "total_steps": TOTAL_STEPS,
            "time_step_minutes": TIME_STEP_MINUTES,
            "start_time": START_TIME.isoformat(),
            "timestamp": timestamp,
        }
    }
    
    with open(result_file, "wb") as f:
        pickle.dump(save_data, f)
    
    logger.info(f"  ✓ 结果已保存到: {result_file}")
    
    # 同时保存为JSON格式以便查看
    json_file = os.path.join(output_dir, f"daily_mobility_results_{timestamp}.json")
    json_data = {
        "results": {
            "gyration_radius": results["gyration_radius"].tolist(),
            "daily_location_numbers": results["daily_location_numbers"].tolist(),
        },
        "metadata": save_data["metadata"]
    }
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"  ✓ JSON格式结果已保存到: {json_file}")

    await society.close()

async def main_social(
    logger,
    num_agents: int = 1,
    profile_start_idx: int = 0,
):
    """
    运行 DailyMobility Benchmark

    实验设置：
    - 模拟起点：当日早上 00:00:00 (UTC)
    - 时间步长：15 分钟 = 900 秒
    - 总步数：97 步（覆盖 24+ 小时）
    """
    logger.info("\n" + "=" * 80)
    logger.info("【DailyMobility Benchmark】")
    logger.info("=" * 80)
    logger.info("实验设置：")
    logger.info("  - 起始时间: 当日早上 00:00:00 (UTC)")
    logger.info("  - 时间步长: 15 分钟 (900 秒)")
    logger.info("  - 总步数: 97 步 (覆盖 7:00 - 23:15)")
    logger.info(f"  - Agent 数量: {num_agents}")
    logger.info("=" * 80)

    # 实验参数
    # 从早上 7 点开始模拟
    START_TIME = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
    TIME_STEP_MINUTES = 15  # 15 分钟
    TIME_STEP_SECONDS = TIME_STEP_MINUTES * 60  # 900 秒
    TOTAL_STEPS = 97

    # 用于存储需要清理的环境
    mobility_env = None
    env_router = None
    agents = []

    # ==================== 加载 Profiles ====================
    logger.info("\n【步骤1/4】加载 profiles.json...")

    profiles_path = os.path.join(os.path.dirname(__file__), "profiles.json")
    if not os.path.exists(profiles_path):
        logger.error(f"  ❌ profiles.json 文件不存在: {profiles_path}")
        return

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    logger.info(f"  ✓ 加载了 {len(profiles)} 个 agent profiles")

    # 限制 agent 数量
    if num_agents > len(profiles):
        logger.warning(
            f"  ⚠ 请求的 agent 数量 ({num_agents}) 超过 profiles 数量 ({len(profiles)})，使用全部 {len(profiles)} 个"
        )
        num_agents = len(profiles)

    profiles_to_use = profiles[profile_start_idx : profile_start_idx + num_agents]

    # 【关键修复】动态获取实际的 agent_ids，而不是硬编码 1-num_agents
    actual_agent_ids = [p["id"] for p in profiles_to_use]
    logger.info(f"  ✓ 实际 Agent IDs: {actual_agent_ids}")

    # ==================== 初始化环境 ====================
    logger.info("\n【步骤2/4】初始化环境...")

    # 清空并创建 Agent 特定的chroma_memories目录
    chroma_base_dir = "/tmp/chroma_memories"
    if os.path.exists(chroma_base_dir):
        shutil.rmtree(chroma_base_dir)
    os.makedirs(chroma_base_dir, exist_ok=True)

    # ==================== 创建 Agents ====================
    logger.info(f"\n【步骤3/4】创建 {num_agents} 个 Agents...")

    agent_args = []
    mobility_persons = []
    for profile in profiles_to_use:
        agent_id = profile["id"]

        # 为每个 agent 创建独立的 chroma 路径
        agent_chroma_path = os.path.join(chroma_base_dir, f"agent_{agent_id}")
        agent_sqlite_path = os.path.join(chroma_base_dir, f"agent_{agent_id}.db")
        os.makedirs(agent_chroma_path, exist_ok=True)

        # 创建 Agent 特定的 memory 配置
        agent_memory_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"agent_{agent_id}_memories",
                    "path": agent_chroma_path,
                    "embedding_model_dims": 1024,
                },
            },
             "storage_config": {
                "provider": "sqlite",
                "path": agent_sqlite_path,
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen2.5-14b-instruct",
                    "api_key": os.getenv("INFINI_API_KEY"),
                    "openai_base_url": "https://cloud.infini-ai.com/maas/v1",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "bge-m3",
                    "api_key": os.getenv("INFINI_API_KEY"),
                    "openai_base_url": "https://cloud.infini-ai.com/maas/v1",
                    "embedding_dims": 1024,
                },
            },
        }

        # 创建 agent（使用 profile 中的详细信息）
        # 构建个人资料字符串
        profile_text = f"My name is Agent-{agent_id}, age {profile.get('age', 30)}, gender {profile.get('gender', 'Unknown')}, education {profile.get('education', 'Unknown')}, occupation {profile.get('occupation', 'Unknown')}, home at {profile['home']}, work at {profile['work']}"

        agent_args.append(
            {
                "id": agent_id,
                "profile": profile_text,
                "memory_config": agent_memory_config,
                "world_description": "",
            }
        )
        mobility_persons.append(
            {
                "id": agent_id,
                "position": {
                    "kind": "aoi",
                    "aoi_id": profile["home"],
                },
            }
        )

    # 创建 MobilitySpace 环境
    # 使用相对路径而不是硬编码的 /root 路径
    home_dir = os.path.join(os.path.expanduser("~"), "simframework_data")
    map_path = os.path.join(home_dir, "beijing.pb")
    os.makedirs(home_dir, exist_ok=True)

    social_env = SimpleSocialSpace(
        agent_id_name_pairs=[
            (agent_id, profile.get("name", f"Agent-{agent_id}"))
            for agent_id, profile in zip(actual_agent_ids, profiles_to_use)
        ]
    )
    # # 创建 DailySpace 环境（使用实际的 agent_ids）
    # daily_env = DailySpace(person_ids=actual_agent_ids)

    # 创建 CodeGenRouter
    env_router = CodeGenRouter(env_modules=[social_env])

    # 生成世界描述
    world_description = await env_router.generate_world_description_from_tools()

    print("--------------------------------")
    print(world_description)
    print("--------------------------------")

    # 更新 agent_args 中的 world_description
    for args in agent_args:
        args["world_description"] = world_description

    # 实际初始化agents
    agents = [PersonAgent(**args) for args in agent_args]

    society = SimFramework(
        agents=agents,
        env_router=env_router,
        start_t=START_TIME,
    )
    await society.init()

    await society.run(num_steps=TOTAL_STEPS, tick=TIME_STEP_SECONDS)

    await society.close()


if __name__ == "__main__":
    setup_logging(
        log_file=f"logs/daily_mobility_benchmark-{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
        log_level=logging.DEBUG,
    )
    asyncio.run(main(logger=get_logger()))
