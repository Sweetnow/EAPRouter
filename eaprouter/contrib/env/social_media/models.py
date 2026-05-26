from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class User(BaseModel):
    """
    User Model
    """
    
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    bio: Optional[str] = Field(None, description="User biography")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation time")
    followers_count: int = Field(0, ge=0, description="Number of followers")
    following_count: int = Field(0, ge=0, description="Number of users being followed")
    posts_count: int = Field(0, ge=0, description="Number of posts created")
    camp_score: Optional[float] = Field(
        None,
        description="Camp score for polarization experiment, optional",
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"User {self.username} (ID: {self.user_id}), Followers: {self.followers_count}, Following: {self.following_count}, Posts: {self.posts_count}"


class Post(BaseModel):
    """
    贴文模型(原创、转发或评论)
    """
    
    post_id: int = Field(..., description="Post ID")
    author_id: int = Field(..., description="Author user ID")
    content: str = Field(..., min_length=1, max_length=5000, description="Post content")
    post_type: Literal["original", "repost", "comment"] = Field("original", description="Post type: original, repost, or comment")
    parent_id: Optional[int] = Field(None, description="Parent post ID (for repost and comment)")
    created_at: datetime = Field(default_factory=datetime.now, description="Post creation time")
    likes_count: int = Field(0, ge=0, description="Number of likes")
    reposts_count: int = Field(0, ge=0, description="Number of reposts")
    comments_count: int = Field(0, ge=0, description="Number of comments")
    view_count: int = Field(0, ge=0, description="Number of views")
    tags: List[str] = Field(default_factory=list, description="话题标签列表，最多10个")
    topic_category: Optional[str] = Field(None, description="主要话题分类（politics/sports/tech等）")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.post_type.capitalize()} Post (ID: {self.post_id}) by User {self.author_id}: {self.content[:50]}{'...' if len(self.content) > 50 else ''}, Likes: {self.likes_count}, Reposts: {self.reposts_count}, Comments: {self.comments_count}"


class Comment(BaseModel):
    """Comment Model"""
    
    comment_id: int = Field(..., description="Comment ID")
    post_id: int = Field(..., description="Post ID that this comment belongs to")
    author_id: int = Field(..., description="Commenter user ID")
    content: str = Field(..., min_length=1, max_length=2000, description="Comment content")
    parent_comment_id: Optional[int] = Field(None, description="Parent comment ID (for replies)")
    created_at: datetime = Field(default_factory=datetime.now, description="Comment creation time")
    likes_count: int = Field(0, ge=0, description="Number of likes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        reply_str = f" (Reply to comment {self.parent_comment_id})" if self.parent_comment_id else ""
        return f"Comment (ID: {self.comment_id}) by User {self.author_id}{reply_str}: {self.content[:30]}{'...' if len(self.content) > 30 else ''}"


class DirectMessage(BaseModel):
    """
    私聊消息模型
    """
    
    message_id: int = Field(..., description="Message ID")
    from_user_id: int = Field(..., description="Sender user ID")
    to_user_id: int = Field(..., description="Receiver user ID")
    content: str = Field(..., min_length=1, max_length=2000, description="Message content")
    created_at: datetime = Field(default_factory=datetime.now, description="Message send time")
    read: bool = Field(False, description="Whether the message has been read")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        read_str = "Read" if self.read else "Unread"
        return f"Direct Message ({read_str}) from User {self.from_user_id} to User {self.to_user_id}: {self.content[:30]}{'...' if len(self.content) > 30 else ''}"


class GroupChat(BaseModel):
    """
    群聊模型
    """
    
    group_id: int = Field(..., description="Group chat ID")
    group_name: str = Field(..., description="Group chat name")
    owner_id: int = Field(..., description="Group owner user ID")
    member_ids: List[int] = Field(default_factory=list, description="List of member user IDs")
    created_at: datetime = Field(default_factory=datetime.now, description="Group creation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Group Chat '{self.group_name}' (ID: {self.group_id}), Owner: {self.owner_id}, Members: {len(self.member_ids)}"


class GroupMessage(BaseModel):
    """
    群聊消息模型
    """
    
    message_id: int = Field(..., description="Message ID")
    group_id: int = Field(..., description="Group chat ID")
    from_user_id: int = Field(..., description="Sender user ID")
    content: str = Field(..., min_length=1, max_length=2000, description="Message content")
    created_at: datetime = Field(default_factory=datetime.now, description="Message send time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Group Message from User {self.from_user_id} in Group {self.group_id}: {self.content[:30]}{'...' if len(self.content) > 30 else ''}"


__all__ = [
    "User",
    "Post",
    "Comment",
    "DirectMessage",
    "GroupChat",
    "GroupMessage",
    # Response Models
    "CreatePostResponse",
    "LikePostResponse",
    "UnlikePostResponse",
    "FollowUserResponse",
    "UnfollowUserResponse",
    "ViewPostResponse",
    "GetUserProfileResponse",
    "GetUserPostsResponse",
    "CommentOnPostResponse",
    "ReplyToCommentResponse",
    "RepostResponse",
    "SendDirectMessageResponse",
    "GetDirectMessagesResponse",
    "CreateGroupChatResponse",
    "SendGroupMessageResponse",
    "GetGroupMessagesResponse",
    "RefreshFeedResponse",
    "SearchPostsResponse",
    "GetTrendingTopicsResponse",
    "GetEnvironmentStatsResponse",
    "GetTopicAnalyticsResponse",
    "ObserveUserResponse",
]


# ============ Response Models ============

class CreatePostResponse(BaseModel):
    """创建帖子的响应"""
    post_id: int = Field(..., description="新创建的帖子ID")
    author_id: int = Field(..., description="作者ID")
    content: str = Field(..., description="帖子内容")
    tags: List[str] = Field(default_factory=list, description="话题标签")
    created_at: str = Field(..., description="创建时间(ISO格式)")
    post_type: str = Field("original", description="帖子类型")


class LikePostResponse(BaseModel):
    """点赞帖子的响应"""
    post_id: int = Field(..., description="帖子ID")
    user_id: int = Field(..., description="点赞用户ID")
    total_likes: int = Field(..., description="帖子当前总点赞数")


class UnlikePostResponse(BaseModel):
    """取消点赞的响应"""
    post_id: int = Field(..., description="帖子ID")
    user_id: int = Field(..., description="用户ID")
    total_likes: int = Field(..., description="帖子当前总点赞数")


class FollowUserResponse(BaseModel):
    """关注用户的响应"""
    follower_id: int = Field(..., description="关注者ID")
    followee_id: int = Field(..., description="被关注者ID")
    follower_following_count: int = Field(..., description="关注者的关注数")
    followee_followers_count: int = Field(..., description="被关注者的粉丝数")


class UnfollowUserResponse(BaseModel):
    """取消关注的响应"""
    follower_id: int = Field(..., description="关注者ID")
    followee_id: int = Field(..., description="被关注者ID")
    follower_following_count: int = Field(..., description="关注者的关注数")
    followee_followers_count: int = Field(..., description="被关注者的粉丝数")


class ViewPostResponse(BaseModel):
    """查看帖子的响应"""
    post_id: int = Field(..., description="帖子ID")
    author_id: int = Field(..., description="作者ID")
    content: str = Field(..., description="帖子内容")
    post_type: str = Field(..., description="帖子类型")
    likes_count: int = Field(..., description="点赞数")
    comments_count: int = Field(..., description="评论数")
    reposts_count: int = Field(..., description="转发数")
    view_count: int = Field(..., description="浏览数")
    created_at: str = Field(..., description="创建时间")
    tags: List[str] = Field(default_factory=list, description="话题标签列表")
    topic_category: Optional[str] = Field(None, description="主要话题分类")


class GetUserProfileResponse(BaseModel):
    """获取用户资料的响应"""
    user_id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    bio: Optional[str] = Field(None, description="用户简介")
    followers_count: int = Field(..., description="粉丝数")
    following_count: int = Field(..., description="关注数")
    posts_count: int = Field(..., description="帖子数")
    recent_posts: List[dict] = Field(default_factory=list, description="最近帖子")


class GetUserPostsResponse(BaseModel):
    """获取用户帖子列表的响应"""
    user_id: int = Field(..., description="用户ID")
    posts: List[dict] = Field(default_factory=list, description="帖子列表")
    count: int = Field(..., description="返回的帖子数量")
    total: int = Field(..., description="用户帖子总数")


class CommentOnPostResponse(BaseModel):
    """评论帖子的响应"""
    comment_id: int = Field(..., description="评论ID")
    post_id: int = Field(..., description="帖子ID")
    user_id: int = Field(..., description="评论者ID")
    content: str = Field(..., description="评论内容")
    total_comments: int = Field(..., description="帖子当前总评论数")


class ReplyToCommentResponse(BaseModel):
    """回复评论的响应"""
    new_comment_id: int = Field(..., description="新评论ID")
    parent_comment_id: int = Field(..., description="父评论ID")
    post_id: int = Field(..., description="帖子ID")
    user_id: int = Field(..., description="回复者ID")
    content: str = Field(..., description="回复内容")


class RepostResponse(BaseModel):
    """转发帖子的响应"""
    new_post_id: int = Field(..., description="新帖子ID")
    original_post_id: int = Field(..., description="原帖子ID")
    user_id: int = Field(..., description="转发者ID")
    comment: str = Field("", description="转发评论")
    original_reposts_count: int = Field(..., description="原帖当前转发数")


class SendDirectMessageResponse(BaseModel):
    """发送私信的响应"""
    message_id: int = Field(..., description="消息ID")
    from_user_id: int = Field(..., description="发送者ID")
    to_user_id: int = Field(..., description="接收者ID")
    content: str = Field(..., description="消息内容")


class GetDirectMessagesResponse(BaseModel):
    """获取私信的响应"""
    user1_id: int = Field(..., description="用户1 ID")
    user2_id: int = Field(..., description="用户2 ID")
    messages: List[dict] = Field(default_factory=list, description="消息列表")
    count: int = Field(..., description="返回的消息数量")
    total: int = Field(..., description="消息总数")
    unread_count: int = Field(..., description="未读消息数")


class CreateGroupChatResponse(BaseModel):
    """创建群聊的响应"""
    group_id: int = Field(..., description="群聊ID")
    group_name: str = Field(..., description="群聊名称")
    owner_id: int = Field(..., description="群主ID")
    member_ids: List[int] = Field(default_factory=list, description="成员ID列表")
    member_count: int = Field(..., description="成员数量")


class SendGroupMessageResponse(BaseModel):
    """发送群消息的响应"""
    message_id: int = Field(..., description="消息ID")
    group_id: int = Field(..., description="群聊ID")
    from_user_id: int = Field(..., description="发送者ID")
    content: str = Field(..., description="消息内容")
    group_name: str = Field(..., description="群聊名称")


class GetGroupMessagesResponse(BaseModel):
    """获取群消息的响应"""
    group_id: int = Field(..., description="群聊ID")
    group_name: str = Field(..., description="群聊名称")
    messages: List[dict] = Field(default_factory=list, description="消息列表")
    count: int = Field(..., description="返回的消息数量")
    total: int = Field(..., description="消息总数")


class RefreshFeedResponse(BaseModel):
    """刷新Feed的响应"""
    user_id: int = Field(..., description="用户ID")
    algorithm: str = Field(..., description="推荐算法")
    posts: List[dict] = Field(default_factory=list, description="推荐帖子列表")
    count: int = Field(..., description="返回的帖子数量")


class SearchPostsResponse(BaseModel):
    """搜索帖子的响应"""
    keyword: str = Field(..., description="搜索关键词")
    tags: List[str] = Field(default_factory=list, description="标签过滤")
    sort_by: str = Field(..., description="排序方式")
    posts: List[dict] = Field(default_factory=list, description="匹配的帖子")
    count: int = Field(..., description="返回的帖子数量")
    total_matched: int = Field(..., description="总匹配数")


class TrendingTopic(BaseModel):
    """热门话题项"""
    topic: str = Field(..., description="话题名称")
    post_count: int = Field(..., description="帖子数量")
    total_interactions: int = Field(..., description="总互动数")
    heat_score: int = Field(..., description="热度分数")


class GetTrendingTopicsResponse(BaseModel):
    """获取热门话题的响应"""
    time_window_hours: int = Field(..., description="时间窗口(小时)")
    topics: List[TrendingTopic] = Field(default_factory=list, description="热门话题列表")
    count: int = Field(..., description="话题数量")


class GetEnvironmentStatsResponse(BaseModel):
    """获取环境统计的响应"""
    total_users: int = Field(..., description="总用户数")
    total_posts: int = Field(..., description="总帖子数")
    total_comments: int = Field(..., description="总评论数")
    total_groups: int = Field(..., description="总群聊数")
    active_users_24h: int = Field(..., description="24小时活跃用户数")
    posts_24h: int = Field(..., description="24小时帖子数")
    current_time: str = Field(..., description="当前时间")
    total_likes: int = Field(..., description="总点赞数")
    total_follows: int = Field(..., description="总关注数")
    avg_followers_per_user: float = Field(..., description="平均粉丝数")
    avg_posts_per_user: float = Field(..., description="平均帖子数")
    time_series: Optional[List[dict]] = Field(None, description="时间序列数据")


class GetTopicAnalyticsResponse(BaseModel):
    """获取话题分析的响应"""
    topic: str = Field(..., description="话题名称")
    time_window_hours: int = Field(..., description="时间窗口(小时)")
    total_posts: int = Field(..., description="帖子总数")
    unique_participants: int = Field(..., description="独立参与者数")
    total_likes: int = Field(..., description="总点赞数")
    total_comments: int = Field(..., description="总评论数")
    total_reposts: int = Field(..., description="总转发数")
    engagement_rate: float = Field(..., description="互动率")
    hourly_distribution: List[dict] = Field(default_factory=list, description="每小时分布")
    top_contributors: List[dict] = Field(default_factory=list, description="Top贡献者")


class ObserveUserResponse(BaseModel):
    """用户观察响应 - 用于 <observe> 指令"""
    user_id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    followers_count: int = Field(0, description="粉丝数")
    following_count: int = Field(0, description="关注数")
    posts_count: int = Field(0, description="帖子数")
    unread_messages_count: int = Field(0, description="未读私信数")
    recent_feed: List[dict] = Field(default_factory=list, description="最近的 Feed 帖子")
    recent_messages: List[dict] = Field(default_factory=list, description="最近的私信")
    available_actions: List[str] = Field(default_factory=list, description="可用的行为")

