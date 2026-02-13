"""
Social Space Environment
This environment provides social communication functionalities for agents.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from simframework.env import (
    EnvBase,
    tool,
)
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message model for social space"""

    sender_id: int
    content: str
    timestamp: datetime
    message_id: int  # Unique identifier for each message
    group_id: Optional[int] = None  # Group ID if this is a group message


class Group(BaseModel):
    """Group model for social space"""

    group_id: int
    name: str
    members: List[int]  # List of agent IDs


# Response models for tool functions
class SendMessageResponse(BaseModel):
    """Response model for send_message() function"""
    sender_id: int = Field(..., description="The ID of the sender agent")
    receiver_id: int = Field(..., description="The ID of the receiver agent")
    content: str = Field(..., description="The content of the message")


class ReceiveMessagesResponse(BaseModel):
    """Response model for receive_messages() function"""
    agent_id: int = Field(..., description="The ID of the agent")
    messages: List[dict] = Field(..., description="List of messages received")


class CreateGroupResponse(BaseModel):
    """Response model for create_group() function"""
    creator_id: int = Field(..., description="The ID of the creator agent")
    group_id: int = Field(..., description="The ID of the created group")
    name: str = Field(..., description="The name of the group")


class JoinGroupResponse(BaseModel):
    """Response model for join_group() function"""
    agent_id: int = Field(..., description="The ID of the agent")
    group_id: int = Field(..., description="The ID of the group")


class LeaveGroupResponse(BaseModel):
    """Response model for leave_group() function"""
    agent_id: int = Field(..., description="The ID of the agent")
    group_id: int = Field(..., description="The ID of the group")


class SendGroupMessageResponse(BaseModel):
    """Response model for send_group_message() function"""
    sender_id: int = Field(..., description="The ID of the sender agent")
    group_id: int = Field(..., description="The ID of the group")
    content: str = Field(..., description="The content of the message")
    recipient_count: int = Field(..., description="The number of recipients")


class SimpleSocialSpace(EnvBase):
    def __init__(
        self,
        agent_id_name_pairs: List[Tuple[int, str]] | List[List[int | str]],
    ):
        """
        Initialize the Social Space environment.

        Args:
            agent_id_name_pairs: List of (agent_id, name) tuples or list of [agent_id, name] lists.
                                  Can be tuples or lists of length 2.
        """
        super().__init__()

        # Convert list format to tuple format if needed
        pairs: List[Tuple[int, str]] = []
        for pair in agent_id_name_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pairs.append((int(pair[0]), str(pair[1])))
            else:
                raise ValueError(
                    f"Invalid agent_id_name_pair format: {pair}. Expected tuple/list of (int, str)"
                )

        # Individual mailboxes for agents
        self._mailboxes: Dict[int, List[Message]] = defaultdict(list)

        # Groups
        self._groups: Dict[int, Group] = {}
        self._next_group_id: int = 1
        self._next_message_id: int = 1

        # Names for agents and groups
        self._agent_names: Dict[int, str] = {agent_id: name for agent_id, name in pairs}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """

        description = f"""{cls.__name__}: Social communication environment module.

**Description:** Provides social communication functionalities including individual mailboxes and group messaging.

**Initialization Parameters (excluding llm):**
- agent_id_name_pairs (List[Tuple[int, str]] | List[List[int | str]]): List of (agent_id, name) pairs. Can be tuples or lists of length 2. Each element should be [agent_id (int), name (str)].

**Example initialization config:**
```json
{{
  "agent_id_name_pairs": [
    [1, "Alice"],
    [2, "Bob"],
    [3, "Charlie"]
  ]
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module for router selection and function calling"""
        return """You are a social communication environment module specialized in managing social interactions between agents.
        
Your task is to use the available social functions to manage individual mailboxes and group communications based on the context provided."""

    # Mailbox functions
    @tool(readonly=False)
    async def send_message(
        self,
        sender_id: int,
        receiver_id: int,
        content: str,
    ) -> SendMessageResponse:
        """
        Send a message to an agent's mailbox.

        Args:
            sender_id: The ID of the sender agent
            receiver_id: The ID of the receiver agent
            content: The content of the message

        Returns:
            Context containing message details
        """
        async with self._lock:
            # Create message
            timestamp = self.t
            message = Message(
                sender_id=sender_id,
                content=content,
                timestamp=timestamp,
                message_id=self._next_message_id,
            )
            self._next_message_id += 1

            # Add to receiver's mailbox
            self._mailboxes[receiver_id].append(message)

            return SendMessageResponse(
                sender_id=sender_id,
                receiver_id=receiver_id,
                content=content,
            )

    @tool(readonly=True, kind="observe")
    async def receive_messages(self, agent_id: int) -> ReceiveMessagesResponse:
        """
        Receive all messages from an agent's mailbox including personal messages and group messages.
        This is like an instant messaging app where all messages are received in one place.

        Args:
            agent_id: The ID of the agent

        Returns:
            Context containing all messages
        """
        async with self._lock:
            # Collect all messages (personal and group)
            messages = []

            # Get personal messages
            if agent_id in self._mailboxes:
                for message in self._mailboxes[agent_id]:
                    # Get sender name if available
                    sender_name = self._agent_names.get(
                        message.sender_id, f"Agent {message.sender_id}"
                    )
                    message_data = {
                        "type": "personal",
                        "sender_id": message.sender_id,
                        "sender_name": sender_name,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                    }
                    # Add group info if it's a group message
                    if message.group_id is not None:
                        message_data["type"] = "group"
                        message_data["group_id"] = message.group_id
                        group_name = (
                            self._groups[message.group_id].name
                            if message.group_id in self._groups
                            else f"Group {message.group_id}"
                        )
                        message_data["group_name"] = group_name
                    messages.append(message_data)
                # Clear personal mailbox
                self._mailboxes[agent_id] = []

            return ReceiveMessagesResponse(
                agent_id=agent_id,
                messages=messages,
            )

    # Group functions
    @tool(readonly=False)
    async def create_group(
        self, creator_id: int, name: str, init_members: List[int]
    ) -> CreateGroupResponse:
        """
        Create a new group.

        Args:
            creator_id: The ID of the agent creating the group
            name: The name of the group
            init_members: The initial members of the group

        Returns:
            Context containing group details
        """
        async with self._lock:
            group_id = self._next_group_id
            self._next_group_id += 1

            group = Group(
                group_id=group_id,
                name=name,
                members=[creator_id] + init_members,
            )

            self._groups[group_id] = group

            return CreateGroupResponse(
                creator_id=creator_id,
                group_id=group_id,
                name=name,
            )

    @tool(readonly=False)
    async def join_group(self, agent_id: int, group_id: int) -> JoinGroupResponse:
        """
        Join an agent to a group.

        Args:
            agent_id: The ID of the agent
            group_id: The ID of the group

        Returns:
            Context containing group and agent details
        """
        async with self._lock:
            # Check if group exists
            if group_id not in self._groups:
                return JoinGroupResponse(agent_id=agent_id, group_id=group_id)

            # Check if agent is already in the group
            group = self._groups[group_id]
            if agent_id in group.members:
                return JoinGroupResponse(agent_id=agent_id, group_id=group_id)

            # Add agent to group
            group.members.append(agent_id)

            return JoinGroupResponse(agent_id=agent_id, group_id=group_id)

    @tool(readonly=False)
    async def leave_group(self, agent_id: int, group_id: int) -> LeaveGroupResponse:
        """
        Remove an agent from a group.

        Args:
            agent_id: The ID of the agent
            group_id: The ID of the group

        Returns:
            Context containing group and agent details
        """
        async with self._lock:
            # Check if group exists
            if group_id not in self._groups:
                return LeaveGroupResponse(agent_id=agent_id, group_id=group_id)

            # Check if agent is in the group
            group = self._groups[group_id]
            if agent_id not in group.members:
                return LeaveGroupResponse(agent_id=agent_id, group_id=group_id)

            # Remove agent from group
            group.members.remove(agent_id)

            # If group is empty, remove it
            if len(group.members) == 0:
                del self._groups[group_id]

            return LeaveGroupResponse(agent_id=agent_id, group_id=group_id)

    @tool(readonly=False)
    async def send_group_message(
        self, sender_id: int, group_id: int, content: str
    ) -> SendGroupMessageResponse:
        """
        Send a message to all members of a group. The message is forwarded to each member's mailbox.

        Args:
            sender_id: The ID of the sender agent
            group_id: The ID of the group
            content: The content of the message

        Returns:
            Context containing message details
        """
        async with self._lock:
            # Check if group exists
            if group_id not in self._groups:
                return SendGroupMessageResponse(
                    sender_id=sender_id,
                    group_id=group_id,
                    content=content,
                    recipient_count=0,
                )

            # Check if sender is in the group
            group = self._groups[group_id]
            if sender_id not in group.members:
                return SendGroupMessageResponse(
                    sender_id=sender_id,
                    group_id=group_id,
                    content=content,
                    recipient_count=0,
                )

            # Create message
            timestamp = self.t
            message = Message(
                sender_id=sender_id,
                content=content,
                timestamp=timestamp,
                message_id=self._next_message_id,
                group_id=group_id,
            )
            self._next_message_id += 1

            # Forward message to each member's mailbox (including sender)
            recipient_count = 0
            for member_id in group.members:
                self._mailboxes[member_id].append(message)
                recipient_count += 1

            return SendGroupMessageResponse(
                sender_id=sender_id,
                group_id=group_id,
                content=content,
                recipient_count=recipient_count,
            )

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        Args:
            tick: The number of ticks of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.t = t

            # Clean up empty groups
            groups_to_remove = []
            for group_id, group in self._groups.items():
                # If group is empty (no members), mark for removal
                if len(group.members) == 0:
                    groups_to_remove.append(group_id)

            # Remove empty groups
            for group_id in groups_to_remove:
                del self._groups[group_id]
