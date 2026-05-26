import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

from eaprouter.env import EnvBase, tool
from eaprouter.logger import get_logger

from .models import (
    User, Post, Comment, DirectMessage, GroupChat, GroupMessage,
    CreatePostResponse, LikePostResponse, UnlikePostResponse,
    FollowUserResponse, UnfollowUserResponse, ViewPostResponse,
    GetUserProfileResponse, GetUserPostsResponse, CommentOnPostResponse,
    ReplyToCommentResponse, RepostResponse, SendDirectMessageResponse,
    GetDirectMessagesResponse, CreateGroupChatResponse, SendGroupMessageResponse,
    GetGroupMessagesResponse, RefreshFeedResponse, SearchPostsResponse,
    GetTrendingTopicsResponse, GetEnvironmentStatsResponse, GetTopicAnalyticsResponse,
    TrendingTopic, ObserveUserResponse,
)
from .recommend import RecommendationEngine
from .schemas import ALL_SOCIAL_SCHEMAS



class SocialMediaSpace(EnvBase):
    """
    Social Media Environment Module (e.g. Weibo/Twitter style).

    Agent 与社交媒体用户的对应关系：
    - 默认（未传 agent_id_name_pairs）：约定 agent_id === user_id。observe_user(person_id) 及
      各 tool 的 user_id 即仿真中的 agent id；用户来自 users.json 或按需自动创建（_ensure_user_exists）。
    - 若传入 agent_id_name_pairs：显式列出参与本环境的 (agent_id, name)。
      此时仅允许这些 id 作为 user_id 使用；init() 时会为列表中尚未在 users 数据里的 id 创建对应用户（username=name）。
    """

    def __init__(
        self,
        users: Optional[Dict[int, Any]] = None,
        posts: Optional[Dict[int, Any]] = None,
        comments: Optional[Dict[int, List[Any]]] = None,
        follows: Optional[Dict[int, List[int]]] = None,
        likes: Optional[Dict[int, List[int]]] = None,
        agent_id_name_pairs: Optional[
            List[Tuple[int, str]] | List[List[Union[int, str]]]
        ] = None,
        **kwargs: Any,
    ):
        """
        初始化社交媒体空间环境。

        Args:
            users: 初始用户，key=user_id, value=User 可序列化 dict。
            posts: 初始帖子，key=post_id, value=Post 可序列化 dict。
            comments: 初始评论，key=post_id, value=该帖下的 Comment dict 列表。
            follows: 关注关系，key=user_id, value=被关注者 user_id 列表。
            likes: 点赞关系，key=user_id, value=被点赞的 post_id 列表。
            agent_id_name_pairs: 可选。显式 agent–用户映射 [(agent_id, name), ...]。
            **kwargs: feed_source, polarization_mode 等实验参数。
        """
        super().__init__()
        self._initial_users = users
        self._initial_posts = posts
        self._initial_comments = comments
        self._initial_follows = follows
        self._initial_likes = likes

        # 极化实验参数（feed 候选池与同阵营/异阵营比例）
        self._feed_source: str = str(kwargs.get("feed_source", "global"))
        self._polarization_mode: str = str(kwargs.get("polarization_mode", "none"))
        self._within_community_ratio: float = float(kwargs.get("within_community_ratio", 0.5))
        self._community_detection: str = str(kwargs.get("community_detection", "follow_components"))
        _seed = kwargs.get("random_seed")
        self._random_seed: Optional[int] = int(_seed) if _seed is not None else None

        # 并发锁，保护状态修改操作
        self._lock = asyncio.Lock()

        self._users: Dict[int, User] = {}
        self._posts: Dict[int, Post] = {}
        self._follows: Dict[int, List[int]] = defaultdict(list)
        self._likes: Dict[int, List[int]] = defaultdict(list)
        self._comments: Dict[int, List[Comment]] = defaultdict(list)
        self._groups: Dict[int, GroupChat] = {}
        self._direct_messages: Dict[str, List[DirectMessage]] = {}
        self._group_messages: Dict[int, List[GroupMessage]] = defaultdict(list)

        self._next_post_id: int = 1
        self._next_comment_id: int = 1
        self._next_group_id: int = 1
        self._next_dm_id: int = 1
        self._next_group_msg_id: int = 1

        # 贴文推荐引擎（Feed Recommendation）；可选预训练模型路径与算法名
        self._rec_engine = RecommendationEngine(
            model_path=kwargs.get("recommendation_model_path"),
            recommendation_algorithm=kwargs.get("recommendation_algorithm", "mf"),
        )

        # 话题索引：tag -> [post_ids]，用于快速搜索
        self._topic_index: Dict[str, List[int]] = defaultdict(list)

        # Event to synchronize table registration
        self._tables_registered = asyncio.Event()

        # Step counter for replay (aligned with agent step; incremented at end of env.step())
        self._step_counter: int = 0
        # Replay id counters for social_like and social_follow (each event needs unique id)
        self._like_replay_id: int = 0
        self._follow_replay_id: int = 0

        # 显式 agent–用户映射：仅允许这些 id 作为 user_id
        self._allowed_user_ids: Optional[Set[int]] = None
        self._agent_names: Dict[int, str] = {}
        if agent_id_name_pairs:
            pairs: List[Tuple[int, str]] = []
            for pair in agent_id_name_pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    pairs.append((int(pair[0]), str(pair[1])))
                else:
                    raise ValueError(
                        f"Invalid agent_id_name_pair: {pair}. Expected (int, str) or [int, str]"
                    )
            self._allowed_user_ids = {aid for aid, _ in pairs}
            self._agent_names = {aid: name for aid, name in pairs}

        get_logger().info("SocialMediaSpace initialized (in-memory data only)")

    async def _wait_for_tables(self) -> None:
        """Wait for tables to be registered."""
        if self._replay_writer is not None:
            await self._tables_registered.wait()

    def _get_community_labels(self) -> Dict[int, int]:
        """
        为每个用户分配社区标签 0 或 1，用于极化实验。
        """
        user_ids = set(self._users.keys())
        if not user_ids:
            return {}
        # 若所有用户都有 camp_score，则直接用阵营分数
        if all(
            getattr(self._users.get(uid), "camp_score", None) is not None
            for uid in user_ids
        ):
            labels: Dict[int, int] = {}
            for uid in user_ids:
                s = getattr(self._users[uid], "camp_score", None)
                labels[uid] = 0 if s is not None and s < 0.5 else 1
            return labels
        # 回退：parity 或 follow_components
        if self._community_detection == "parity":
            return {uid: int(uid % 2) for uid in user_ids}
        adj: Dict[int, List[int]] = defaultdict(list)
        for uid in user_ids:
            for followee in self._follows.get(uid, []):
                if followee in user_ids:
                    adj[uid].append(followee)
                    adj[followee].append(uid)
        visited: Dict[int, bool] = {}
        components_list: List[List[int]] = []
        for uid in user_ids:
            if visited.get(uid):
                continue
            comp: List[int] = []
            stack = [uid]
            while stack:
                u = stack.pop()
                if visited.get(u):
                    continue
                visited[u] = True
                comp.append(u)
                for v in adj.get(u, []):
                    if not visited.get(v):
                        stack.append(v)
            if comp:
                components_list.append(comp)
        components_list.sort(key=len, reverse=True)
        labels = {}
        for i, comp in enumerate(components_list):
            cid = 0 if i == 0 else 1
            for u in comp:
                labels[u] = cid
        return labels

    def _get_candidate_posts(self, user_id: int) -> List[Post]:
        """按 feed_source 得到候选帖子列表：global=全站，following=仅关注者+自己的帖子。"""
        all_posts = list(self._posts.values())
        if self._feed_source != "following":
            return all_posts
        followees = set(self._follows.get(user_id, []))
        allow_authors = followees | {user_id}
        return [p for p in all_posts if p.author_id in allow_authors]

    def _apply_polarization_mix(
        self, user_id: int, candidate_posts: List[Post], limit: int
    ) -> List[Post]:
        """
        当 polarization_mode=="follow_community" 时，按 within_community_ratio
        从同阵营与异阵营作者中混合取样，再按时间倒序；否则直接返回 candidate_posts。
        """
        if self._polarization_mode != "follow_community" or not candidate_posts:
            return candidate_posts
        labels = self._get_community_labels()
        viewer_community = labels.get(user_id, 0)
        same: List[Post] = [p for p in candidate_posts if labels.get(p.author_id, 0) == viewer_community]
        other: List[Post] = [p for p in candidate_posts if labels.get(p.author_id, 0) != viewer_community]
        rng = random.Random(self._random_seed if self._random_seed is not None else 0)
        n_same = max(0, int(round(self._within_community_ratio * limit)))
        n_other = limit - n_same
        if n_same >= len(same) and n_other >= len(other):
            mixed = same + other
        else:
            shuffled_same = list(same)
            shuffled_other = list(other)
            rng.shuffle(shuffled_same)
            rng.shuffle(shuffled_other)
            mixed = shuffled_same[:n_same] + shuffled_other[:n_other]
        mixed.sort(key=lambda p: p.created_at, reverse=True)
        return mixed[:limit]

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Used by workspace init to generate .EAPRouter/env_modules/social_media.json.
        """
        user_schema = User.model_json_schema()
        post_schema = Post.model_json_schema()
        comment_schema = Comment.model_json_schema()
        description = f"""{cls.__name__}: Social media platform environment module.

**Description:** Full-featured social media: posts, likes, follows, comments, DMs, group chats, feed recommendations.

**Initialization – pass initial data in memory:**
- users (dict, optional): Map user_id (int) -> User-like dict (user_id, username, bio, created_at ISO string, followers_count, following_count, posts_count, optional camp_score).
- posts (dict, optional): Map post_id (int) -> Post-like dict (post_id, author_id, content, post_type "original"|"repost"|"comment", parent_id, created_at ISO, likes_count, reposts_count, comments_count, view_count, tags, topic_category).
- comments (dict, optional): Map post_id (int) -> list of Comment-like dicts (comment_id, post_id, author_id, content, parent_comment_id, created_at ISO, likes_count).
- follows (dict, optional): Map user_id (int) -> list of followed user_id (int).
- likes (dict, optional): Map user_id (int) -> list of liked post_id (int).
- feed_source ("global" | "following", optional): "global" = all posts; "following" = only followees + self. Default: "global".
- polarization_mode ("none" | "follow_community", optional): Default: "none".
- within_community_ratio (float), community_detection ("follow_components" | "parity"), random_seed (int): Optional.
- agent_id_name_pairs (list of [agent_id, name]): Explicit agent–user mapping; users not in initial data are created with the given name.

**Initial data (example) :**

User (each key = user_id):
```json
{json.dumps(user_schema, indent=2)}
```

Post (each key = post_id):
```json
{json.dumps(post_schema, indent=2)}
```

Comment (each key = post_id, value = list of comments):
```json
{json.dumps(comment_schema, indent=2)}
```

Example payloads (ISO datetimes, keys may be int or string in JSON; constructor accepts both):
- users: {{ 1: {{ "user_id": 1, "username": "alice", "bio": null, "created_at": "2024-08-01T00:00:00+00:00", "followers_count": 0, "following_count": 0, "posts_count": 0 }}, ... }}
- posts: {{ 1: {{ "post_id": 1, "author_id": 1, "content": "Hello.", "post_type": "original", "parent_id": null, "created_at": "2024-08-10T12:00:00+00:00", "likes_count": 0, "reposts_count": 0, "comments_count": 0, "view_count": 0, "tags": [], "topic_category": null }}, ... }}
- comments: {{ 1: [ {{ "comment_id": 1, "post_id": 1, "author_id": 2, "content": "A comment.", "parent_comment_id": null, "created_at": "2024-08-10T12:30:00+00:00", "likes_count": 0 }} ] }}, ... }}
- follows: {{ 1: [2, 3], 2: [1] }}
- likes: {{ 1: [1, 2], 2: [1] }}

**Example initialization config:**
```json
{{
  "users": {{ "1": {{ "user_id": 1, "username": "alice", "created_at": "2024-08-01T00:00:00" }}, "2": {{ "user_id": 2, "username": "bob", "created_at": "2024-08-01T00:00:00" }} }},
  "posts": {{ "1": {{ "post_id": 1, "author_id": 1, "content": "First post.", "post_type": "original", "created_at": "2024-08-10T12:00:00" }} }},
  "comments": {{}},
  "follows": {{ "1": [2], "2": [1] }},
  "likes": {{}},
  "feed_source": "global",
  "polarization_mode": "none"
}}
```
"""
        return description
    
    @property
    def description(self) -> str:
        """Description of the environment module for router selection and function calling"""
        return """You are a social media platform environment module specialized in managing social media operations.

Your task is to use the available tools to:
- Create and view posts (original posts, reposts, comments)
- Like/unlike posts
- Follow/unfollow users
- Send direct messages and group chats
- Generate personalized feeds with recommendation algorithms

Use the available tools based on the agent's request."""

    @staticmethod
    def _norm_user_data(data: Any) -> Dict[str, Any]:
        """Normalize user dict for User(...); accept ISO datetime strings."""
        d = dict(data)
        if "created_at" in d and isinstance(d["created_at"], str):
            d["created_at"] = datetime.fromisoformat(d["created_at"].replace("Z", "+00:00"))
        return d

    @staticmethod
    def _norm_post_data(data: Any) -> Dict[str, Any]:
        """Normalize post dict for Post(...)."""
        d = dict(data)
        if "created_at" in d and isinstance(d["created_at"], str):
            d["created_at"] = datetime.fromisoformat(d["created_at"].replace("Z", "+00:00"))
        return d

    @staticmethod
    def _norm_comment_data(data: Any) -> Dict[str, Any]:
        """Normalize comment dict for Comment(...)."""
        d = dict(data)
        if "created_at" in d and isinstance(d["created_at"], str):
            d["created_at"] = datetime.fromisoformat(d["created_at"].replace("Z", "+00:00"))
        return d

    def _apply_initial_data(self) -> None:
        """Populate _users, _posts, _comments, _follows, _likes from constructor initial data."""
        self._users = {}
        for uid, data in (self._initial_users or {}).items():
            self._users[int(uid)] = User(**self._norm_user_data(data))
        self._posts = {}
        for pid, data in (self._initial_posts or {}).items():
            self._posts[int(pid)] = Post(**self._norm_post_data(data))
        self._follows = defaultdict(list)
        for uid, followee_ids in (self._initial_follows or {}).items():
            self._follows[int(uid)] = [int(x) for x in followee_ids]
        self._likes = defaultdict(list)
        for uid, post_ids in (self._initial_likes or {}).items():
            self._likes[int(uid)] = [int(x) for x in post_ids]
        self._comments = defaultdict(list)
        for post_id, comment_list in (self._initial_comments or {}).items():
            self._comments[int(post_id)] = [
                Comment(**self._norm_comment_data(c)) for c in comment_list
            ]
        if self._posts:
            self._next_post_id = max(self._posts.keys()) + 1
        else:
            self._next_post_id = 1
        all_comment_ids = [
            c.comment_id for comments in self._comments.values() for c in comments
        ]
        self._next_comment_id = max(all_comment_ids) + 1 if all_comment_ids else 1
        get_logger().info(
            f"Applied initial data: {len(self._users)} users, {len(self._posts)} posts, "
            f"{len(self._follows)} follows, {len(self._comments)} post comments"
        )

    async def init(self, start_datetime: datetime):
        """
        Initialize the environment module. Uses in-memory initial data (users, posts, ...) if provided; otherwise starts with empty state. Persistence is handled by external DB.
        """
        self.t = start_datetime
        has_initial = any(
            x is not None
            for x in (
                self._initial_users,
                self._initial_posts,
                self._initial_comments,
                self._initial_follows,
                self._initial_likes,
            )
        )
        if has_initial:
            self._apply_initial_data()
        # 未传初始数据时保持空状态，持久化由外部数据库负责

        # 显式映射时：为 agent_id_name_pairs 中尚未存在的 id 创建对应用户
        if self._allowed_user_ids is not None:
            for aid in self._allowed_user_ids:
                if aid not in self._users:
                    name = self._agent_names.get(aid, f"user_{aid}")
                    self._users[aid] = User(user_id=aid, username=name)
                    get_logger().info(f"Created user for agent {aid} (username={name})")

        # Register social media tables if replay writer is available
        if self._replay_writer is not None:
            for schema in ALL_SOCIAL_SCHEMAS:
                await self._replay_writer.register_table(schema)
            self._tables_registered.set()
            get_logger().info("Registered all social media tables for SocialMediaSpace")
    
    async def step(self, tick: int, t: datetime):
        """
        Run forward one step
        
        Args:
            tick: Number of ticks of this simulation step
            t: Current datetime after this step
        """
        self.t = t
        # Social media doesn't need per-step updates
        # All updates happen through @tool method calls
    
    async def close(self):
        """Close the environment module. Data persistence is handled by external DB."""
        get_logger().info("SocialMediaSpace closed")

    def set_replay_writer(self, writer) -> None:
        super().set_replay_writer(writer)
        self._schedule_replay_task(self._sync_replay_state())
    
    def _schedule_replay_task(self, coro) -> None:
        if self._replay_writer is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(coro)

    async def _sync_replay_state(self) -> None:
        if self._replay_writer is None:
            return
            
        # Ensure tables are registered before writing data
        # This handles the case where set_replay_writer is called before init()
        for schema in ALL_SOCIAL_SCHEMAS:
            await self._replay_writer.register_table(schema)
        self._tables_registered.set()
        get_logger().info("Registered social media tables during sync_replay_state")

        for user in self._users.values():
            await self._write_social_user(user)
        for post in self._posts.values():
            await self._write_social_post(post)
        for comment_list in self._comments.values():
            for comment in comment_list:
                await self._write_social_comment(comment)
        for follower_id, followees in self._follows.items():
            for followee_id in followees:
                await self._write_social_follow_event(
                    follower_id=follower_id,
                    followee_id=followee_id,
                    action="follow",
                    created_at=None,
                )
        for post_id, likes in self._likes.items():
            for user_id in likes:
                await self._write_social_like_event(
                    post_id=post_id,
                    user_id=user_id,
                    action="like",
                    created_at=None,
                )
        for group in self._groups.values():
            await self._write_social_group(group)
        for dm_list in self._direct_messages.values():
            for dm in dm_list:
                await self._write_social_dm(dm)
        for gm_list in self._group_messages.values():
            for message in gm_list:
                await self._write_social_group_message(message)

    async def _write_social_user(self, user: User) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        profile = user.model_dump(mode="json")
        profile["agent_id"] = user.user_id
        await self._replay_writer.write("social_user", {
            "user_id": user.user_id,
            "username": user.username,
            "bio": user.bio,
            "created_at": user.created_at,
            "followers_count": user.followers_count,
            "following_count": user.following_count,
            "posts_count": user.posts_count,
            "profile": profile,
        })

    async def _write_social_post(self, post: Post) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        await self._replay_writer.write("social_post", {
            "post_id": post.post_id,
            "step": self._step_counter,
            "author_id": post.author_id,
            "content": post.content,
            "post_type": post.post_type,
            "parent_id": post.parent_id,
            "created_at": post.created_at,
            "likes_count": post.likes_count,
            "reposts_count": post.reposts_count,
            "comments_count": post.comments_count,
            "view_count": post.view_count,
            "tags": post.tags,
            "topic_category": post.topic_category,
        })

    async def _write_social_comment(self, comment: Comment) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        await self._replay_writer.write("social_comment", {
            "comment_id": comment.comment_id,
            "step": self._step_counter,
            "post_id": comment.post_id,
            "author_id": comment.author_id,
            "content": comment.content,
            "parent_comment_id": comment.parent_comment_id,
            "created_at": comment.created_at,
            "likes_count": comment.likes_count,
        })

    async def _write_social_follow_event(
        self,
        follower_id: int,
        followee_id: int,
        action: str,
        created_at: Optional[datetime],
    ) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        self._follow_replay_id += 1
        await self._replay_writer.write("social_follow", {
            "id": self._follow_replay_id,
            "step": self._step_counter,
            "follower_id": follower_id,
            "followee_id": followee_id,
            "action": action,
            "created_at": created_at,
        })

    async def _write_social_like_event(
        self,
        post_id: int,
        user_id: int,
        action: str,
        created_at: Optional[datetime],
    ) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        self._like_replay_id += 1
        await self._replay_writer.write("social_like", {
            "id": self._like_replay_id,
            "step": self._step_counter,
            "post_id": post_id,
            "user_id": user_id,
            "action": action,
            "created_at": created_at,
        })

    async def _write_social_dm(self, message: DirectMessage) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        await self._replay_writer.write("social_dm", {
            "message_id": message.message_id,
            "step": self._step_counter,
            "from_user_id": message.from_user_id,
            "to_user_id": message.to_user_id,
            "content": message.content,
            "created_at": message.created_at,
            "read": 1 if message.read else 0,
        })

    async def _write_social_group(self, group: GroupChat) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        await self._replay_writer.write("social_group", {
            "group_id": group.group_id,
            "group_name": group.group_name,
            "owner_id": group.owner_id,
            "member_ids": group.member_ids,
            "created_at": group.created_at,
        })

    async def _write_social_group_message(self, message: GroupMessage) -> None:
        if self._replay_writer is None:
            return
        await self._wait_for_tables()
        await self._replay_writer.write("social_group_message", {
            "message_id": message.message_id,
            "step": self._step_counter,
            "group_id": message.group_id,
            "from_user_id": message.from_user_id,
            "content": message.content,
            "created_at": message.created_at,
        })
        
    def _dump_state(self) -> dict:
        """
        Dump internal state（包含新增字段）
        """
        state = {
            "users": {uid: user.model_dump() for uid, user in self._users.items()},
            "posts": {pid: post.model_dump() for pid, post in self._posts.items()},
            "follows": dict(self._follows),
            "likes": dict(self._likes),
            "comments": {
                pid: [c.model_dump() for c in comment_list]
                for pid, comment_list in self._comments.items()
            },
            "groups": {gid: group.model_dump() for gid, group in self._groups.items()},
            "direct_messages": {
                key: [dm.model_dump() for dm in dm_list]
                for key, dm_list in self._direct_messages.items()
            },
            "group_messages": {
                gid: [gm.model_dump() for gm in gm_list]
                for gid, gm_list in self._group_messages.items()
            },
            "next_post_id": self._next_post_id,
            "next_comment_id": self._next_comment_id,
            "next_group_id": self._next_group_id,
            "next_dm_id": self._next_dm_id,
            "next_group_msg_id": self._next_group_msg_id,
            
            # 新增字段
            "topic_index": dict(self._topic_index),  # 话题索引
        }

        return state
    
    def _load_state(self, state: dict):
        """
        Load internal state（包含新增字段）
        """
        try:
            if "users" in state:
                self._users = {
                    int(uid): User(**data) for uid, data in state["users"].items()
                }
            
            if "posts" in state:
                self._posts = {
                    int(pid): Post(**data) for pid, data in state["posts"].items()
                }
            
            if "follows" in state:
                self._follows = defaultdict(list, {
                    int(k): v for k, v in state["follows"].items()
                })
            
            if "likes" in state:
                self._likes = defaultdict(list, {
                    int(k): v for k, v in state["likes"].items()
                })
            
            if "comments" in state:
                self._comments = defaultdict(list)
                for pid, comment_list in state["comments"].items():
                    self._comments[int(pid)] = [Comment(**c) for c in comment_list]
            
            if "groups" in state:
                self._groups = {
                    int(gid): GroupChat(**data) for gid, data in state["groups"].items()
                }
            
            if "direct_messages" in state:
                self._direct_messages = {}
                for key, dm_list in state["direct_messages"].items():
                    self._direct_messages[key] = [DirectMessage(**dm) for dm in dm_list]
            
            if "group_messages" in state:
                self._group_messages = defaultdict(list)
                for gid, gm_list in state["group_messages"].items():
                    self._group_messages[int(gid)] = [GroupMessage(**gm) for gm in gm_list]
            
            if "next_post_id" in state:
                self._next_post_id = state["next_post_id"]
            if "next_comment_id" in state:
                self._next_comment_id = state["next_comment_id"]
            if "next_group_id" in state:
                self._next_group_id = state["next_group_id"]
            if "next_dm_id" in state:
                self._next_dm_id = state["next_dm_id"]
            if "next_group_msg_id" in state:
                self._next_group_msg_id = state["next_group_msg_id"]
            
            # 加载话题索引
            if "topic_index" in state:
                self._topic_index = defaultdict(list, {
                    k: v for k, v in state["topic_index"].items()
                })

            get_logger().info("State loaded successfully")
        except Exception as e:
            get_logger().error(f"Failed to load state: {e}")
    

    # @tool Methods

    @tool(readonly=True, kind="observe")
    async def observe_user(self, person_id: int) -> ObserveUserResponse:
        """
        观察用户当前状态

        用于 <observe> 指令，返回用户可见的社交媒体环境信息。

        Args:
            person_id: 用户ID

        Returns:
            ObserveUserResponse 响应模型，包含用户状态和可用行为
        """
        user_id = person_id
        self._ensure_user_exists(user_id)
        user = self._users[user_id]

        # 获取最近的 Feed
        candidate_posts = self._get_candidate_posts(user_id)
        if candidate_posts:
            if self._polarization_mode == "follow_community":
                recent_feed_posts = self._apply_polarization_mix(user_id, candidate_posts, 5)
            else:
                recent_feed_posts = self._rec_engine.chronological(candidate_posts, user_id, limit=5)
            recent_feed = [p.model_dump() for p in recent_feed_posts]
        else:
            recent_feed = []

        # 获取未读私信
        unread_count = 0
        recent_messages = []
        for conv_key, dm_list in self._direct_messages.items():
            for dm in dm_list:
                if dm.to_user_id == user_id and not dm.read:
                    unread_count += 1
                    recent_messages.append(dm.model_dump())

        # 限制最近私信数量
        recent_messages = sorted(
            recent_messages,
            key=lambda m: m.get("created_at", ""),
            reverse=True
        )[:5]

        # 可用行为列表
        available_actions = [
            "create_post(author_id, content, tags=[]) - 发布帖子",
            "like_post(user_id, post_id) - 点赞帖子",
            "unlike_post(user_id, post_id) - 取消点赞",
            "follow_user(follower_id, followee_id) - 关注用户",
            "unfollow_user(follower_id, followee_id) - 取消关注",
            "view_post(user_id, post_id) - 查看帖子详情",
            "comment_on_post(user_id, post_id, content) - 评论帖子",
            "repost(user_id, post_id, comment='') - 转发帖子",
            "send_direct_message(from_user_id, to_user_id, content) - 发送私信",
            "refresh_feed(user_id, algorithm='chronological', limit=20) - 刷新Feed",
            "search_posts(keyword, tags=[], limit=20) - 搜索帖子",
            "get_trending_topics(time_window_hours=24) - 获取热门话题",
        ]

        return ObserveUserResponse(
            user_id=user.user_id,
            username=user.username,
            followers_count=user.followers_count,
            following_count=user.following_count,
            posts_count=user.posts_count,
            unread_messages_count=unread_count,
            recent_feed=recent_feed,
            recent_messages=recent_messages,
            available_actions=available_actions
        )

    @tool(readonly=False)
    async def create_post(
        self,
        author_id: int,
        content: str,
        tags: List[str] = []
    ) -> CreatePostResponse:
        """
        Create a new original post (支持话题标签)

        Args:
            author_id: ID of the author
            content: Content of the post
            tags: 话题标签列表，例如 ["guncontrol", "politics"]

        Returns:
            CreatePostResponse with post details
        """
        async with self._lock:
            self._ensure_user_exists(author_id)

            post_id = self._get_next_post_id()
            post = Post(
                post_id=post_id,
                author_id=author_id,
                content=content,
                tags=tags,
                post_type="original",
                created_at=self.t
            )

            self._posts[post_id] = post
            self._users[author_id].posts_count += 1

            for tag in tags:
                self._topic_index[tag].append(post_id)

            get_logger().info(f"User {author_id} created post {post_id} with tags {tags}")

            await self._write_social_post(post)
            await self._write_social_user(self._users[author_id])

            return CreatePostResponse(
                post_id=post_id,
                author_id=author_id,
                content=content,
                tags=tags,
                created_at=post.created_at.isoformat(),
                post_type="original"
            )
    
    @tool(readonly=False)
    async def like_post(
        self,
        user_id: int,
        post_id: int
    ) -> LikePostResponse:
        """
        Like a post

        Args:
            user_id: ID of the user who likes
            post_id: ID of the post to like

        Returns:
            LikePostResponse with like details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            if post_id not in self._posts:
                raise ValueError(f"Post {post_id} does not exist")

            if user_id in self._likes[post_id]:
                raise ValueError(f"User {user_id} has already liked post {post_id}")

            self._likes[post_id].append(user_id)
            self._posts[post_id].likes_count += 1

            get_logger().info(f"User {user_id} liked post {post_id}")

            await self._write_social_like_event(
                post_id=post_id,
                user_id=user_id,
                action="like",
                created_at=self.t,
            )
            await self._write_social_post(self._posts[post_id])

            return LikePostResponse(
                post_id=post_id,
                user_id=user_id,
                total_likes=self._posts[post_id].likes_count
            )
    
    @tool(readonly=False)
    async def unlike_post(
        self,
        user_id: int,
        post_id: int
    ) -> UnlikePostResponse:
        """
        Unlike a post

        Args:
            user_id: ID of the user who unlikes
            post_id: ID of the post to unlike

        Returns:
            UnlikePostResponse with unlike details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            if post_id not in self._posts:
                raise ValueError(f"Post {post_id} does not exist")

            if user_id not in self._likes[post_id]:
                raise ValueError(f"User {user_id} has not liked post {post_id}")

            self._likes[post_id].remove(user_id)
            self._posts[post_id].likes_count -= 1

            get_logger().info(f"User {user_id} unliked post {post_id}")

            await self._write_social_like_event(
                post_id=post_id,
                user_id=user_id,
                action="unlike",
                created_at=self.t,
            )
            await self._write_social_post(self._posts[post_id])

            return UnlikePostResponse(
                post_id=post_id,
                user_id=user_id,
                total_likes=self._posts[post_id].likes_count
            )

    @tool(readonly=False)
    async def follow_user(
        self,
        follower_id: int,
        followee_id: int
    ) -> FollowUserResponse:
        """
        Follow a user

        Args:
            follower_id: ID of the follower
            followee_id: ID of the user to follow

        Returns:
            FollowUserResponse with follow details
        """
        async with self._lock:
            self._ensure_user_exists(follower_id)
            self._ensure_user_exists(followee_id)

            if follower_id == followee_id:
                raise ValueError(f"Failed to follow: user {follower_id} cannot follow themselves")

            if followee_id in self._follows[follower_id]:
                raise ValueError(f"User {follower_id} is already following user {followee_id}")

            self._follows[follower_id].append(followee_id)
            self._users[follower_id].following_count += 1
            self._users[followee_id].followers_count += 1

            get_logger().info(f"User {follower_id} followed user {followee_id}")

            await self._write_social_follow_event(
                follower_id=follower_id,
                followee_id=followee_id,
                action="follow",
                created_at=self.t,
            )
            await self._write_social_user(self._users[follower_id])
            await self._write_social_user(self._users[followee_id])

            return FollowUserResponse(
                follower_id=follower_id,
                followee_id=followee_id,
                follower_following_count=self._users[follower_id].following_count,
                followee_followers_count=self._users[followee_id].followers_count
            )

    @tool(readonly=False)
    async def unfollow_user(
        self,
        follower_id: int,
        followee_id: int
    ) -> UnfollowUserResponse:
        """
        Unfollow a user

        Args:
            follower_id: ID of the follower
            followee_id: ID of the user to unfollow

        Returns:
            UnfollowUserResponse with unfollow details
        """
        async with self._lock:
            self._ensure_user_exists(follower_id)
            self._ensure_user_exists(followee_id)

            if followee_id not in self._follows[follower_id]:
                raise ValueError(f"User {follower_id} is not following user {followee_id}")

            self._follows[follower_id].remove(followee_id)
            self._users[follower_id].following_count -= 1
            self._users[followee_id].followers_count -= 1

            get_logger().info(f"User {follower_id} unfollowed user {followee_id}")

            await self._write_social_follow_event(
                follower_id=follower_id,
                followee_id=followee_id,
                action="unfollow",
                created_at=self.t,
            )
            await self._write_social_user(self._users[follower_id])
            await self._write_social_user(self._users[followee_id])

            return UnfollowUserResponse(
                follower_id=follower_id,
                followee_id=followee_id,
                follower_following_count=self._users[follower_id].following_count,
                followee_followers_count=self._users[followee_id].followers_count
            )
    
    @tool(readonly=False)
    async def view_post(
        self,
        user_id: int,
        post_id: int
    ) -> ViewPostResponse:
        """
        View a post (increments view count)

        Args:
            user_id: ID of the user viewing
            post_id: ID of the post to view

        Returns:
            ViewPostResponse with post details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            if post_id not in self._posts:
                raise ValueError(f"Failed to view: post {post_id} does not exist")

            post = self._posts[post_id]
            post.view_count += 1

            get_logger().debug(f"User {user_id} viewed post {post_id}")

            return ViewPostResponse(
                post_id=post.post_id,
                author_id=post.author_id,
                content=post.content,
                post_type=post.post_type,
                likes_count=post.likes_count,
                comments_count=post.comments_count,
                reposts_count=post.reposts_count,
                view_count=post.view_count,
                created_at=post.created_at.isoformat(),
                tags=post.tags,
                topic_category=post.topic_category,
            )
    
    @tool(readonly=True)
    async def get_user_profile(
        self,
        user_id: int
    ) -> GetUserProfileResponse:
        """
        Get user profile information
        
        Args:
            user_id: ID of the user
            
        Returns:
            Tuple of (context_dict, answer_string)
        """
        if user_id not in self._users:
            raise ValueError(f"User {user_id} does not exist")
        
        user = self._users[user_id]
        
        # 获取用户最新的 5 条帖子（暂定）
        user_posts = [
            post for post in self._posts.values()
            if post.author_id == user_id
        ]
        user_posts.sort(key=lambda p: p.created_at, reverse=True)
        recent_posts = user_posts[:5]
        
        return GetUserProfileResponse(
            user_id=user.user_id,
            username=user.username,
            bio=user.bio,
            followers_count=user.followers_count,
            following_count=user.following_count,
            posts_count=user.posts_count,
            recent_posts=[p.model_dump() for p in recent_posts]
        )
    
    @tool(readonly=True)
    async def get_user_posts(
        self,
        user_id: int,
        limit: int = 20
    ) -> GetUserPostsResponse:
        """
        Get posts created by a user
        
        Args:
            user_id: ID of the user
            limit: Maximum number of posts to return
            
        Returns:
            Tuple of (context_dict, answer_string)
        """
        if user_id not in self._users:
            raise ValueError(f"User {user_id} does not exist")
        
        # 获取用户的所有帖子
        user_posts = [
            post for post in self._posts.values()
            if post.author_id == user_id
        ]
        
        # 根据发布时间降序排列
        user_posts.sort(key=lambda p: p.created_at, reverse=True)
        
        limited_posts = user_posts[:limit]
        
        return GetUserPostsResponse(
            user_id=user_id,
            posts=[p.model_dump() for p in limited_posts],
            count=len(limited_posts),
            total=len(user_posts)
        )
    
    @tool(readonly=False)
    async def comment_on_post(
        self,
        user_id: int,
        post_id: int,
        content: str
    ) -> CommentOnPostResponse:
        """
        Comment on a post

        Args:
            user_id: ID of the commenter
            post_id: ID of the post to comment on
            content: Comment content

        Returns:
            CommentOnPostResponse with comment details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            if post_id not in self._posts:
                raise ValueError(f"Failed to comment: post {post_id} does not exist")

            comment_id = self._next_comment_id
            self._next_comment_id += 1

            comment = Comment(
                comment_id=comment_id,
                post_id=post_id,
                author_id=user_id,
                content=content,
                created_at=self.t
            )

            self._comments[post_id].append(comment)
            self._posts[post_id].comments_count += 1

            get_logger().info(f"User {user_id} commented on post {post_id}")

            await self._write_social_comment(comment)
            await self._write_social_post(self._posts[post_id])

            return CommentOnPostResponse(
                comment_id=comment_id,
                post_id=post_id,
                user_id=user_id,
                content=content,
                total_comments=self._posts[post_id].comments_count
            )

    @tool(readonly=False)
    async def reply_to_comment(
        self,
        user_id: int,
        comment_id: int,
        content: str
    ) -> ReplyToCommentResponse:
        """
        Reply to a comment

        Args:
            user_id: ID of the replier
            comment_id: ID of the comment to reply to
            content: Reply content

        Returns:
            ReplyToCommentResponse with reply details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            parent_comment = None
            parent_post_id = None
            for post_id, comment_list in self._comments.items():
                for comment in comment_list:
                    if comment.comment_id == comment_id:
                        parent_comment = comment
                        parent_post_id = post_id
                        break
                if parent_comment:
                    break

            if not parent_comment:
                raise ValueError(f"Failed to reply: comment {comment_id} does not exist")

            new_comment_id = self._next_comment_id
            self._next_comment_id += 1

            reply = Comment(
                comment_id=new_comment_id,
                post_id=parent_post_id,
                author_id=user_id,
                content=content,
                parent_comment_id=comment_id,
                created_at=self.t
            )

            self._comments[parent_post_id].append(reply)
            self._posts[parent_post_id].comments_count += 1

            get_logger().info(f"User {user_id} replied to comment {comment_id}")

            await self._write_social_comment(reply)
            await self._write_social_post(self._posts[parent_post_id])

            return ReplyToCommentResponse(
                new_comment_id=new_comment_id,
                parent_comment_id=comment_id,
                post_id=parent_post_id,
                user_id=user_id,
                content=content
            )
    
    @tool(readonly=False)
    async def repost(
        self,
        user_id: int,
        post_id: int,
        comment: str = ""
    ) -> RepostResponse:
        """
        Repost a post (with optional comment)

        Args:
            user_id: ID of the user reposting
            post_id: ID of the post to repost
            comment: Optional comment on the repost

        Returns:
            RepostResponse with repost details
        """
        async with self._lock:
            self._ensure_user_exists(user_id)

            if post_id not in self._posts:
                raise ValueError(f"Failed to repost: post {post_id} does not exist")

            new_post_id = self._get_next_post_id()

            repost_content = comment if comment else f"repost {post_id}"

            repost_post = Post(
                post_id=new_post_id,
                author_id=user_id,
                content=repost_content,
                post_type="repost",
                parent_id=post_id,
                created_at=self.t
            )

            self._posts[new_post_id] = repost_post
            self._posts[post_id].reposts_count += 1
            self._users[user_id].posts_count += 1

            get_logger().info(f"User {user_id} reposted post {post_id} as {new_post_id}")

            await self._write_social_post(repost_post)
            await self._write_social_post(self._posts[post_id])
            await self._write_social_user(self._users[user_id])

            return RepostResponse(
                new_post_id=new_post_id,
                original_post_id=post_id,
                user_id=user_id,
                comment=comment,
                original_reposts_count=self._posts[post_id].reposts_count
            )

    @tool(readonly=False)
    async def send_direct_message(
        self,
        from_user_id: int,
        to_user_id: int,
        content: str
    ) -> SendDirectMessageResponse:
        """
        Send a direct message to another user

        Args:
            from_user_id: ID of the sender
            to_user_id: ID of the receiver
            content: Message content

        Returns:
            SendDirectMessageResponse with message details
        """
        async with self._lock:
            self._ensure_user_exists(from_user_id)
            self._ensure_user_exists(to_user_id)

            if from_user_id == to_user_id:
                raise ValueError(
                    f"Failed to send message: user {from_user_id} cannot message themselves"
                )

            message_id = self._next_dm_id
            self._next_dm_id += 1

            dm = DirectMessage(
                message_id=message_id,
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                content=content,
                created_at=self.t,
                read=False
            )

            conv_key = self._get_dm_key(from_user_id, to_user_id)

            if conv_key not in self._direct_messages:
                self._direct_messages[conv_key] = []

            self._direct_messages[conv_key].append(dm)

            get_logger().info(f"User {from_user_id} sent DM to user {to_user_id}")

            await self._write_social_dm(dm)

            return SendDirectMessageResponse(
                message_id=message_id,
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                content=content
            )
    
    @tool(readonly=True)
    async def get_direct_messages(
        self,
        user1_id: int,
        user2_id: int,
        limit: int = 50
    ) -> GetDirectMessagesResponse:
        """
        Get direct messages between two users
        
        Args:
            user1_id: ID of user 1
            user2_id: ID of user 2
            limit: Maximum number of messages to return
            
        Returns:
            Tuple of (context_dict, answer_string)
        """
        self._ensure_user_exists(user1_id)
        self._ensure_user_exists(user2_id)
        
        conv_key = self._get_dm_key(user1_id, user2_id)
        messages = self._direct_messages.get(conv_key, [])
        
        sorted_messages = sorted(messages, key=lambda m: m.created_at, reverse=True)
        limited_messages = sorted_messages[:limit]
        
        unread_count = sum(
            1 for m in messages
            if m.to_user_id == user1_id and not m.read
        )
        
        return GetDirectMessagesResponse(
            user1_id=user1_id,
            user2_id=user2_id,
            messages=[m.model_dump() for m in limited_messages],
            count=len(limited_messages),
            total=len(messages),
            unread_count=unread_count
        )
    
    @tool(readonly=False)
    async def create_group_chat(
        self,
        owner_id: int,
        group_name: str,
        member_ids: List[int]
    ) -> CreateGroupChatResponse:
        """
        Create a group chat

        Args:
            owner_id: ID of the group owner
            group_name: Name of the group
            member_ids: List of member IDs (should include owner)

        Returns:
            CreateGroupChatResponse with group details
        """
        async with self._lock:
            self._ensure_user_exists(owner_id)

            for member_id in member_ids:
                self._ensure_user_exists(member_id)

            if owner_id not in member_ids:
                member_ids.append(owner_id)

            group_id = self._next_group_id
            self._next_group_id += 1

            group = GroupChat(
                group_id=group_id,
                group_name=group_name,
                owner_id=owner_id,
                member_ids=member_ids,
                created_at=self.t
            )

            self._groups[group_id] = group

            get_logger().info(f"User {owner_id} created group chat {group_id} with {len(member_ids)} members")

            await self._write_social_group(group)

            return CreateGroupChatResponse(
                group_id=group_id,
                group_name=group_name,
                owner_id=owner_id,
                member_ids=member_ids,
                member_count=len(member_ids)
            )
    
    @tool(readonly=False)
    async def send_group_message(
        self,
        group_id: int,
        from_user_id: int,
        content: str
    ) -> SendGroupMessageResponse:
        """
        Send a message to a group chat

        Args:
            group_id: ID of the group
            from_user_id: ID of the sender
            content: Message content

        Returns:
            SendGroupMessageResponse with message details
        """
        async with self._lock:
            self._ensure_user_exists(from_user_id)

            if group_id not in self._groups:
                raise ValueError(f"Failed to send message: group {group_id} does not exist")

            group = self._groups[group_id]

            if from_user_id not in group.member_ids:
                raise ValueError(
                    f"Failed to send message: user {from_user_id} is not a member of group {group_id}"
                )

            message_id = self._next_group_msg_id
            self._next_group_msg_id += 1

            message = GroupMessage(
                message_id=message_id,
                group_id=group_id,
                from_user_id=from_user_id,
                content=content,
                created_at=self.t
            )

            self._group_messages[group_id].append(message)

            get_logger().info(f"User {from_user_id} sent message to group {group_id}")

            await self._write_social_group_message(message)

            return SendGroupMessageResponse(
                message_id=message_id,
                group_id=group_id,
                from_user_id=from_user_id,
                content=content,
                group_name=group.group_name
            )
    
    @tool(readonly=True)
    async def get_group_messages(
        self,
        group_id: int,
        limit: int = 50
    ) -> GetGroupMessagesResponse:
        """
        Get messages from a group chat
        
        Args:
            group_id: ID of the group
            limit: Maximum number of messages to return
            
        Returns:
            Tuple of (context_dict, answer_string)
        """
        if group_id not in self._groups:
            raise ValueError(f"Failed to get messages: group {group_id} does not exist")
        
        group = self._groups[group_id]
        messages = self._group_messages.get(group_id, [])
        
        sorted_messages = sorted(messages, key=lambda m: m.created_at, reverse=True)
        limited_messages = sorted_messages[:limit]
        
        return GetGroupMessagesResponse(
            group_id=group_id,
            group_name=group.group_name,
            messages=[m.model_dump() for m in limited_messages],
            count=len(limited_messages),
            total=len(messages)
        )
    
    @tool(readonly=True)
    async def refresh_feed(
        self,
        user_id: int,
        algorithm: str = "chronological",
        limit: int = 20
    ) -> RefreshFeedResponse:
        """
        刷新用户Feed流（贴文推荐流 Feed Recommendation）

        **注意**: 这是贴文流推荐,不是物品推荐(Item Recommendation)
        - 贴文推荐: 社交媒体的动态内容流(如Twitter/微博Timeline)
        - 物品推荐: 电商/电影等静态物品推荐(应使用独立的API)

        Args:
            user_id: 用户ID
            algorithm: 贴文推荐算法
                - "chronological": 时间倒序
                - "reddit_hot": Reddit热度排序
                - "twitter_ranking": Twitter综合排序(考虑社交关系)
                - "random": 随机推荐
                - "mf" / "model": 预训练推荐模型（需在构造时传入 recommendation_model_path）
            limit: 返回贴文数量

        Returns:
            (context_dict, answer_string) 元组
        """
        self._ensure_user_exists(user_id)

        # 按 feed_source 得到候选帖子（global=全站，following=关注+自己）
        candidate_posts = self._get_candidate_posts(user_id)

        if not candidate_posts:
            return RefreshFeedResponse(
                user_id=user_id,
                algorithm=algorithm,
                posts=[],
                count=0
            )

        # 极化混合：若 polarization_mode=="follow_community"，按 within_community_ratio 混合同/异阵营
        if self._polarization_mode == "follow_community":
            candidate_posts = self._apply_polarization_mix(user_id, candidate_posts, limit)
            # 混合后已按时间倒序；若算法非 chronological 则再按该算法重排
            if algorithm == "chronological":
                recommended_posts = candidate_posts
            elif algorithm == "reddit_hot":
                recommended_posts = self._rec_engine.reddit_hot(
                    candidate_posts, user_id, limit
                )
            elif algorithm == "twitter_ranking":
                recommended_posts = self._rec_engine.twitter_ranking(
                    candidate_posts,
                    user_id,
                    limit,
                    follows=dict(self._follows),
                    likes=dict(self._likes)
                )
            elif algorithm == "random":
                rng = random.Random(self._random_seed)
                if len(candidate_posts) <= limit:
                    recommended_posts = list(candidate_posts)
                else:
                    recommended_posts = rng.sample(candidate_posts, limit)
            elif algorithm in ("mf", "model") or algorithm == self._rec_engine.get_model_algorithm_name():
                recommended_posts = self._rec_engine.model_recommend(
                    candidate_posts, user_id, limit, exclude_post_ids=None
                )
            else:
                recommended_posts = candidate_posts
        else:
            # 无极化：直接按算法排序
            if algorithm == "chronological":
                recommended_posts = self._rec_engine.chronological(
                    candidate_posts, user_id, limit
                )
            elif algorithm == "reddit_hot":
                recommended_posts = self._rec_engine.reddit_hot(
                    candidate_posts, user_id, limit
                )
            elif algorithm == "twitter_ranking":
                recommended_posts = self._rec_engine.twitter_ranking(
                    candidate_posts,
                    user_id,
                    limit,
                    follows=dict(self._follows),
                    likes=dict(self._likes)
                )
            elif algorithm == "random":
                if self._random_seed is not None:
                    rng = random.Random(self._random_seed)
                    recommended_posts = rng.sample(candidate_posts, limit) if len(candidate_posts) > limit else list(candidate_posts)
                else:
                    recommended_posts = self._rec_engine.random_recommend(
                        candidate_posts, user_id, limit
                    )
            elif algorithm in ("mf", "model") or algorithm == self._rec_engine.get_model_algorithm_name():
                # 预训练推荐模型（如 MF）；未加载模型时 model_recommend 内部回退为时间序
                recommended_posts = self._rec_engine.model_recommend(
                    candidate_posts, user_id, limit, exclude_post_ids=None
                )
            else:
                get_logger().warning(f"Unknown algorithm '{algorithm}', using chronological")
                recommended_posts = self._rec_engine.chronological(
                    candidate_posts, user_id, limit
                )
        
        get_logger().info(
            f"User {user_id} refreshed feed with algorithm '{algorithm}', got {len(recommended_posts)} posts"
        )
        
        return RefreshFeedResponse(
            user_id=user_id,
            algorithm=algorithm,
            posts=[p.model_dump() for p in recommended_posts],
            count=len(recommended_posts)
        )

    @tool(readonly=True)
    async def search_posts(
        self,
        keyword: str,
        tags: List[str] = [],
        limit: int = 20,
        sort_by: str = "time"  # "time", "relevance", "popularity"
    ) -> SearchPostsResponse:
        """
        搜索贴文
        
        Args:
            keyword: 关键词（在content和tags中搜索）
            tags: 指定话题标签过滤
            limit: 返回数量
            sort_by: 排序方式
                - "time": 时间倒序（默认）
                - "relevance": 相关度（关键词出现次数）
                - "popularity": 热度（likes + comments + reposts）
                
        Returns:
            匹配的贴文列表
        """
        keyword_lower = keyword.lower()
        matched_posts = []
        
        # 搜索逻辑
        for post in self._posts.values():
            # 标签过滤
            if tags and not any(tag in post.tags for tag in tags):
                continue
            
            # 关键词匹配
            in_content = keyword_lower in post.content.lower()
            in_tags = any(keyword_lower in tag.lower() for tag in post.tags)
            
            if in_content or in_tags:
                # 计算相关度分数（用于排序）
                relevance_score = 0
                if in_content:
                    relevance_score += post.content.lower().count(keyword_lower)
                if in_tags:
                    relevance_score += 10  # 标签匹配权重高
                
                matched_posts.append({
                    "post": post,
                    "relevance_score": relevance_score,
                    "popularity_score": post.likes_count + post.comments_count * 2 + post.reposts_count * 3
                })
        
        # 排序
        if sort_by == "time":
            matched_posts.sort(key=lambda x: x["post"].created_at, reverse=True)
        elif sort_by == "relevance":
            matched_posts.sort(key=lambda x: x["relevance_score"], reverse=True)
        elif sort_by == "popularity":
            matched_posts.sort(key=lambda x: x["popularity_score"], reverse=True)
        
        # 限制数量
        result_posts = [item["post"] for item in matched_posts[:limit]]
        
        get_logger().info(
            f"Search '{keyword}' with tags {tags}: found {len(matched_posts)} posts, returning {len(result_posts)}"
        )
        
        return SearchPostsResponse(
            keyword=keyword,
            tags=tags,
            sort_by=sort_by,
            posts=[p.model_dump() for p in result_posts],
            count=len(result_posts),
            total_matched=len(matched_posts)
        )

    @tool(readonly=True)
    async def get_trending_topics(
        self,
        time_window_hours: int = 24,
        limit: int = 10
    ) -> GetTrendingTopicsResponse:
        """
        获取热门话题
        
        Args:
            time_window_hours: 时间窗口（小时）
            limit: 返回数量
            
        Returns:
            热门话题列表，按热度排序
        """
        from collections import Counter

        cutoff_time = self.t - timedelta(hours=time_window_hours)
        
        # 统计时间窗口内的所有tags
        recent_tags = []
        for post in self._posts.values():
            if post.created_at >= cutoff_time:
                recent_tags.extend(post.tags)
        
        # 计数并排序
        tag_counts = Counter(recent_tags)
        trending = []
        
        for tag, count in tag_counts.most_common(limit):
            # 计算热度分数（考虑互动）
            tag_posts = [p for p in self._posts.values() if tag in p.tags and p.created_at >= cutoff_time]
            total_interactions = sum(p.likes_count + p.comments_count + p.reposts_count for p in tag_posts)
            
            trending.append({
                "topic": tag,
                "post_count": count,
                "total_interactions": total_interactions,
                "heat_score": count * 10 + total_interactions  # 简单热度公式
            })
        
        # 按热度分数排序
        trending.sort(key=lambda x: x["heat_score"], reverse=True)
        
        get_logger().info(f"Trending topics in last {time_window_hours}h: {[t['topic'] for t in trending]}")
        
        trending_topics = [TrendingTopic(**item) for item in trending]

        return GetTrendingTopicsResponse(
            time_window_hours=time_window_hours,
            topics=trending_topics,
            count=len(trending_topics)
        )

    @tool(readonly=True)
    async def get_environment_stats(
        self,
        include_time_series: bool = False
    ) -> GetEnvironmentStatsResponse:
        """
        获取环境统计信息
        
        Args:
            include_time_series: 是否包含时间序列数据（每小时统计）
            
        Returns:
            环境统计字典
        """
        # 计算活跃用户（最近24小时）
        cutoff_24h = self.t - timedelta(hours=24)
        active_users_24h = set()
        posts_24h = 0
        
        for post in self._posts.values():
            if post.created_at >= cutoff_24h:
                active_users_24h.add(post.author_id)
                posts_24h += 1
        
        for comments in self._comments.values():
            for comment in comments:
                if comment.created_at >= cutoff_24h:
                    active_users_24h.add(comment.author_id)
        
        # 基础统计
        stats = {
            "total_users": len(self._users),
            "total_posts": len(self._posts),
            "total_comments": sum(len(comments) for comments in self._comments.values()),
            "total_groups": len(self._groups),
            "active_users_24h": len(active_users_24h),
            "posts_24h": posts_24h,
            "current_time": self.t.isoformat(),
            
            # 互动统计
            "total_likes": sum(len(likes) for likes in self._likes.values()),
            "total_follows": sum(len(follows) for follows in self._follows.values()),
            
            # 平均值
            "avg_followers_per_user": sum(u.followers_count for u in self._users.values()) / max(len(self._users), 1),
            "avg_posts_per_user": sum(u.posts_count for u in self._users.values()) / max(len(self._users), 1),
        }
        
        # 时间序列（可选）
        if include_time_series:
            stats["time_series"] = self._generate_time_series_stats()
        
        get_logger().info(f"Environment stats: {stats['total_users']} users, {stats['total_posts']} posts, {stats['active_users_24h']} active")
        
        return GetEnvironmentStatsResponse(**stats)
    
    def _generate_time_series_stats(self) -> List[dict]:
        """生成每小时的时间序列统计"""
        from collections import defaultdict

        # 按小时分组
        hourly_stats = defaultdict(lambda: {"posts": 0, "comments": 0, "users": set()})
        
        for post in self._posts.values():
            hour_key = post.created_at.replace(minute=0, second=0, microsecond=0)
            hourly_stats[hour_key]["posts"] += 1
            hourly_stats[hour_key]["users"].add(post.author_id)
        
        for comments in self._comments.values():
            for comment in comments:
                hour_key = comment.created_at.replace(minute=0, second=0, microsecond=0)
                hourly_stats[hour_key]["comments"] += 1
                hourly_stats[hour_key]["users"].add(comment.author_id)
        
        # 转换为列表
        time_series = []
        for hour, data in sorted(hourly_stats.items()):
            time_series.append({
                "timestamp": hour.isoformat(),
                "posts": data["posts"],
                "comments": data["comments"],
                "active_users": len(data["users"])
            })
        
        return time_series

    @tool(readonly=True)
    async def get_topic_analytics(
        self,
        topic: str,
        time_window_hours: int = 24
    ) -> GetTopicAnalyticsResponse:
        """
        获取特定话题的深度分析
                
        Args:
            topic: 话题标签
            time_window_hours: 时间窗口
            
        Returns:
            话题分析数据
        """
        cutoff_time = self.t - timedelta(hours=time_window_hours)
        
        # 筛选相关贴文
        topic_posts = [
            p for p in self._posts.values()
            if topic in p.tags and p.created_at >= cutoff_time
        ]
        
        if not topic_posts:
            raise ValueError(
                f"No posts found for topic '{topic}' in the last {time_window_hours} hours"
            )
        
        # 统计参与用户
        participants = set(p.author_id for p in topic_posts)
        
        # 统计互动
        total_likes = sum(p.likes_count for p in topic_posts)
        total_comments = sum(p.comments_count for p in topic_posts)
        total_reposts = sum(p.reposts_count for p in topic_posts)
        
        # 按时间分布
        hourly_distribution = self._get_hourly_distribution(topic_posts)
        
        # Top贡献者
        author_counts = {}
        for post in topic_posts:
            author_counts[post.author_id] = author_counts.get(post.author_id, 0) + 1
        
        top_contributors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analytics = {
            "topic": topic,
            "time_window_hours": time_window_hours,
            "total_posts": len(topic_posts),
            "unique_participants": len(participants),
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_reposts": total_reposts,
            "engagement_rate": (total_likes + total_comments + total_reposts) / max(len(topic_posts), 1),
            "hourly_distribution": hourly_distribution,
            "top_contributors": [
                {"user_id": uid, "post_count": count}
                for uid, count in top_contributors
            ]
        }
        
        return GetTopicAnalyticsResponse(**analytics)
    
    def _get_hourly_distribution(self, posts: List[Post]) -> List[dict]:
        """计算贴文的每小时分布"""
        from collections import defaultdict
        
        hourly = defaultdict(int)
        for post in posts:
            hour = post.created_at.hour
            hourly[hour] += 1
        
        return [{"hour": h, "count": c} for h, c in sorted(hourly.items())]
    
    async def init_users(self, user_ids: List[int]) -> None:
        """
        Initialize users (helper method, not a @tool)

        Args:
            user_ids: List of user IDs to initialize
        """
        for user_id in user_ids:
            self._ensure_user_exists(user_id)

        get_logger().info(f"Initialized {len(user_ids)} users")

    # 一些辅助函数
    
    def _ensure_user_exists(self, user_id: int) -> None:
        """
        若 user_id 不在当前用户集中则创建对应用户。
        当初始化时传入了 agent_id_name_pairs 时，仅允许该集合内的 id；否则允许任意 id 并按需创建。
        """
        if self._allowed_user_ids is not None and user_id not in self._allowed_user_ids:
            raise ValueError(
                f"User id {user_id} is not in the allowed agent set (agent_id_name_pairs). "
                f"Allowed ids: {sorted(self._allowed_user_ids)}"
            )
        if user_id not in self._users:
            name = self._agent_names.get(user_id, f"user_{user_id}")
            self._users[user_id] = User(user_id=user_id, username=name)
            get_logger().info(f"Auto-created user {user_id} (username={name})")
            self._schedule_replay_task(self._write_social_user(self._users[user_id]))

    def _get_next_post_id(self) -> int:
        """Get next available post ID"""
        post_id = self._next_post_id
        self._next_post_id += 1
        return post_id
    
    def _get_dm_key(self, user1_id: int, user2_id: int) -> str:
        """Get conversation key for direct messages (smaller ID first)"""
        id1, id2 = min(user1_id, user2_id), max(user1_id, user2_id)
        return f"{id1}_{id2}"


__all__ = ["SocialMediaSpace"]
