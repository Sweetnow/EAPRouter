"""Query-only SQLAlchemy Table definitions for replay API.

Social media tables are created and written by SocialMediaSpace (via ReplayWriter).
This module exposes Table() definitions for the replay router to query them,
similar to agent_position in the mobility module.
"""

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text

_meta = MetaData()

# social_user - written by SocialMediaSpace
social_user_table = Table(
    "social_user",
    _meta,
    Column("user_id", Integer, primary_key=True),
    Column("username", String, nullable=False),
    Column("bio", Text),
    Column("created_at", DateTime),
    Column("followers_count", Integer),
    Column("following_count", Integer),
    Column("posts_count", Integer),
    Column("profile", Text),  # JSON stored as TEXT in SQLite
)

# social_post
social_post_table = Table(
    "social_post",
    _meta,
    Column("post_id", Integer, primary_key=True),
    Column("step", Integer, nullable=False),
    Column("author_id", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("post_type", String, nullable=False),
    Column("parent_id", Integer),
    Column("created_at", DateTime),
    Column("likes_count", Integer),
    Column("reposts_count", Integer),
    Column("comments_count", Integer),
    Column("view_count", Integer),
    Column("tags", Text),  # JSON
    Column("topic_category", String),
)

# social_comment
social_comment_table = Table(
    "social_comment",
    _meta,
    Column("comment_id", Integer, primary_key=True),
    Column("step", Integer, nullable=False),
    Column("post_id", Integer, nullable=False),
    Column("author_id", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("parent_comment_id", Integer),
    Column("created_at", DateTime),
    Column("likes_count", Integer),
)

# social_follow
social_follow_table = Table(
    "social_follow",
    _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("step", Integer, nullable=False),
    Column("follower_id", Integer, nullable=False),
    Column("followee_id", Integer, nullable=False),
    Column("action", String, nullable=False),
    Column("created_at", DateTime),
)

# social_dm
social_dm_table = Table(
    "social_dm",
    _meta,
    Column("message_id", Integer, primary_key=True),
    Column("step", Integer, nullable=False),
    Column("from_user_id", Integer, nullable=False),
    Column("to_user_id", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime),
    Column("read", Integer),
)

# social_group
social_group_table = Table(
    "social_group",
    _meta,
    Column("group_id", Integer, primary_key=True),
    Column("group_name", String, nullable=False),
    Column("owner_id", Integer, nullable=False),
    Column("member_ids", Text),  # JSON
    Column("created_at", DateTime),
)

# social_group_message
social_group_message_table = Table(
    "social_group_message",
    _meta,
    Column("message_id", Integer, primary_key=True),
    Column("step", Integer, nullable=False),
    Column("group_id", Integer, nullable=False),
    Column("from_user_id", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime),
)
