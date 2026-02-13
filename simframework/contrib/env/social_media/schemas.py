"""Social media table schema definitions for SocialMediaSpace."""

from simframework.storage import ColumnDef, TableSchema

# Social user table schema
SOCIAL_USER_SCHEMA = TableSchema(
    name="social_user",
    columns=[
        ColumnDef("user_id", "INTEGER", nullable=False),
        ColumnDef("username", "TEXT", nullable=False),
        ColumnDef("bio", "TEXT"),
        ColumnDef("created_at", "TIMESTAMP"),
        ColumnDef("followers_count", "INTEGER"),
        ColumnDef("following_count", "INTEGER"),
        ColumnDef("posts_count", "INTEGER"),
        ColumnDef("profile", "JSON"),
    ],
    primary_key=["user_id"],
)

# Social post table schema (step = simulation step when post was created)
SOCIAL_POST_SCHEMA = TableSchema(
    name="social_post",
    columns=[
        ColumnDef("post_id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("author_id", "INTEGER", nullable=False),
        ColumnDef("content", "TEXT", nullable=False),
        ColumnDef("post_type", "TEXT", nullable=False),
        ColumnDef("parent_id", "INTEGER"),
        ColumnDef("created_at", "TIMESTAMP"),
        ColumnDef("likes_count", "INTEGER"),
        ColumnDef("reposts_count", "INTEGER"),
        ColumnDef("comments_count", "INTEGER"),
        ColumnDef("view_count", "INTEGER"),
        ColumnDef("tags", "JSON"),
        ColumnDef("topic_category", "TEXT"),
    ],
    primary_key=["post_id"],
    indexes=[["author_id"], ["step"], ["created_at"]],
)

# Social comment table schema (step = simulation step when comment was created)
SOCIAL_COMMENT_SCHEMA = TableSchema(
    name="social_comment",
    columns=[
        ColumnDef("comment_id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("post_id", "INTEGER", nullable=False),
        ColumnDef("author_id", "INTEGER", nullable=False),
        ColumnDef("content", "TEXT", nullable=False),
        ColumnDef("parent_comment_id", "INTEGER"),
        ColumnDef("created_at", "TIMESTAMP"),
        ColumnDef("likes_count", "INTEGER"),
    ],
    primary_key=["comment_id"],
    indexes=[["post_id"], ["step"]],
)

# Social follow table schema (step = simulation step when follow event occurred)
SOCIAL_FOLLOW_SCHEMA = TableSchema(
    name="social_follow",
    columns=[
        ColumnDef("id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("follower_id", "INTEGER", nullable=False),
        ColumnDef("followee_id", "INTEGER", nullable=False),
        ColumnDef("action", "TEXT", nullable=False),
        ColumnDef("created_at", "TIMESTAMP"),
    ],
    primary_key=["id"],
    indexes=[["follower_id"], ["followee_id"], ["step"]],
)

# Social like table schema (step = simulation step when like event occurred)
SOCIAL_LIKE_SCHEMA = TableSchema(
    name="social_like",
    columns=[
        ColumnDef("id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("post_id", "INTEGER", nullable=False),
        ColumnDef("user_id", "INTEGER", nullable=False),
        ColumnDef("action", "TEXT", nullable=False),
        ColumnDef("created_at", "TIMESTAMP"),
    ],
    primary_key=["id"],
    indexes=[["post_id"], ["user_id"], ["step"]],
)

# Social direct message table schema (step = simulation step when message was sent)
SOCIAL_DM_SCHEMA = TableSchema(
    name="social_dm",
    columns=[
        ColumnDef("message_id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("from_user_id", "INTEGER", nullable=False),
        ColumnDef("to_user_id", "INTEGER", nullable=False),
        ColumnDef("content", "TEXT", nullable=False),
        ColumnDef("created_at", "TIMESTAMP"),
        ColumnDef("read", "INTEGER"),
    ],
    primary_key=["message_id"],
    indexes=[["from_user_id", "to_user_id"], ["step"], ["to_user_id", "step"]],
)

# Social group table schema
SOCIAL_GROUP_SCHEMA = TableSchema(
    name="social_group",
    columns=[
        ColumnDef("group_id", "INTEGER", nullable=False),
        ColumnDef("group_name", "TEXT", nullable=False),
        ColumnDef("owner_id", "INTEGER", nullable=False),
        ColumnDef("member_ids", "JSON"),
        ColumnDef("created_at", "TIMESTAMP"),
    ],
    primary_key=["group_id"],
)

# Social group message table schema (step = simulation step when message was sent)
SOCIAL_GROUP_MESSAGE_SCHEMA = TableSchema(
    name="social_group_message",
    columns=[
        ColumnDef("message_id", "INTEGER", nullable=False),
        ColumnDef("step", "INTEGER", nullable=False),
        ColumnDef("group_id", "INTEGER", nullable=False),
        ColumnDef("from_user_id", "INTEGER", nullable=False),
        ColumnDef("content", "TEXT", nullable=False),
        ColumnDef("created_at", "TIMESTAMP"),
    ],
    primary_key=["message_id"],
    indexes=[["group_id"], ["step"]],
)

# All social media schemas for convenience
ALL_SOCIAL_SCHEMAS = [
    SOCIAL_USER_SCHEMA,
    SOCIAL_POST_SCHEMA,
    SOCIAL_COMMENT_SCHEMA,
    SOCIAL_FOLLOW_SCHEMA,
    SOCIAL_LIKE_SCHEMA,
    SOCIAL_DM_SCHEMA,
    SOCIAL_GROUP_SCHEMA,
    SOCIAL_GROUP_MESSAGE_SCHEMA,
]
