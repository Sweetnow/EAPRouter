"""Event Space: Event Management and Tracking Module

Supports 7 specific event types:
- sleep: Sleeping/rest periods
- home activity: Activities at home (e.g., cooking, cleaning, household chores)
- work: Work-related activities and tasks
- shopping: Shopping or purchasing activities
- eating out: Eating at restaurants or food establishments
- leisure and entertainment: Leisure, entertainment, social activities
- other: Any other activities not covered above
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from eaprouter.env import EnvBase, tool
from eaprouter.logger import get_logger
from pydantic import BaseModel, ConfigDict, Field

# Valid event types for person behavior tracking
VALID_EVENT_TYPES = Literal[
    "sleep",
    "home activity",
    "other",
    "work",
    "shopping",
    "eating out",
    "leisure and entertainment",
]

DEFAULT_ALLOWED_EVENT_TYPES = [
    "sleep",
    "home activity",
    "other",
    "work",
    "shopping",
    "eating out",
    "leisure and entertainment",
]

# Event types configuration string - easily modifiable for future changes
# This string is used in description to show available event commands to the LLM
DEFAULT_EVENT_TYPES_CONFIG = """
**Sleep Events**: start_event(person_id, "sleep", event_name, duration_seconds)
  - Examples: "night sleep", "afternoon nap"
  - You should not do other activities during sleep. Unless you wake up unexpectedly.

**Home Activity Events**: start_event(person_id, "home activity", event_name, duration_seconds)
  - Examples: "cooking dinner", "cleaning house", "laundry"

**Work Events**: start_event(person_id, "work", event_name, duration_seconds)
  - Examples: "team meeting", "coding task", "email work"

**Shopping Events**: start_event(person_id, "shopping", event_name, duration_seconds)
  - Examples: "grocery shopping", "clothes shopping"

**Eating Out Events**: start_event(person_id, "eating out", event_name, duration_seconds)
  - Examples: "lunch at restaurant", "dinner with friends"

**Leisure and Entertainment Events**: start_event(person_id, "leisure and entertainment", event_name, duration_seconds)
  - Examples: "watching movie", "playing game", "socializing"

**Other Events**: start_event(person_id, "other", event_name, duration_seconds)
  - For any activities not covered by the above categories
"""

__all__ = [
    "EventSpace",
    "CurrentEvent",
    "StartEventResponse",
    "GetEventResponse",
    "EndEventResponse",
    "VALID_EVENT_TYPES",
]


class CurrentEvent(BaseModel):
    """Represents a person's current behavior event"""

    person_id: int = Field(..., description="Person ID")
    event_type: str = Field(
        ...,
        description="Type of event",
    )
    event_name: str = Field(
        ...,
        description="Name/description of the event (e.g., 'afternoon nap', 'meeting with team')",
    )
    status: Literal["in_progress", "completed", "cancelled"] = Field(
        default="in_progress", description="Current status of the event"
    )
    start_time: datetime = Field(..., description="Event start time")
    expected_end_time: Optional[datetime] = Field(
        default=None, description="Expected end time provided by person"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def type(self) -> str:
        """Alias for event_type to support both .type and .event_type access"""
        return self.event_type
    
    @property
    def name(self) -> str:
        """Alias for event_name to support both .name and .event_name access"""
        return self.event_name

    def get_elapsed_seconds(self, current_time: datetime) -> float:
        """Get elapsed time in seconds"""
        return (current_time - self.start_time).total_seconds()

    def get_remaining_seconds(self, current_time: datetime) -> Optional[float]:
        """Get remaining time in seconds until expected end time (if set)"""
        if self.expected_end_time is None or self.status != "in_progress":
            return None
        remaining = self.expected_end_time - current_time
        return max(0, remaining.total_seconds())

    def get_progress_percentage(self, current_time: datetime) -> Optional[float]:
        """Get progress percentage (0-100) based on expected end time"""
        if self.expected_end_time is None:
            return None
        total_duration = (self.expected_end_time - self.start_time).total_seconds()
        if total_duration <= 0:
            return 0
        elapsed = self.get_elapsed_seconds(current_time)
        return min(100, (elapsed / total_duration) * 100)


# Response models for tool functions
class StartEventResponse(BaseModel):
    """Response model for start_event() function"""

    person_id: int = Field(..., description="Person ID")
    event_type: str = Field(..., description="Type of event")
    event_name: str = Field(..., description="Event name/description")
    start_time: datetime = Field(..., description="Event start time")
    expected_end_time: datetime = Field(..., description="Expected end time")
    status: str = Field(..., description="Event status")
    
    @property
    def type(self) -> str:
        """Alias for event_type to support both .type and .event_type access"""
        return self.event_type
    
    @property
    def name(self) -> str:
        """Alias for event_name to support both .name and .event_name access"""
        return self.event_name


class GetEventResponse(BaseModel):
    """Response model for get_event() function"""

    person_id: int = Field(..., description="Person ID")
    event_type: str = Field(..., description="Type of event")
    event_name: str = Field(..., description="Event name/description")
    status: str = Field(..., description="Event status")
    start_time: datetime = Field(..., description="Start time")
    expected_end_time: Optional[datetime] = Field(None, description="Expected end time")
    elapsed_seconds: float = Field(..., description="Elapsed time in seconds")
    remaining_seconds: Optional[float] = Field(
        None, description="Remaining time in seconds (if expected_end_time set)"
    )
    progress_percentage: Optional[float] = Field(
        None, description="Progress percentage (0-100, if expected_end_time set)"
    )
    
    @property
    def type(self) -> str:
        """Alias for event_type to support both .type and .event_type access"""
        return self.event_type
    
    @property
    def name(self) -> str:
        """Alias for event_name to support both .name and .event_name access"""
        return self.event_name


class EndEventResponse(BaseModel):
    """Response model for end_event() function"""

    success: bool = Field(..., description="Whether update was successful")
    message: str = Field(..., description="Status message")


class EventSpace(EnvBase):
    """
    Event Space: A module for managing and tracking person's current behavior event.

    Each person maintains at most one active event at a time.

    Features:
    - Start an event for a person with expected duration
    - Track event progress and duration
    - Query event status and metrics
    - End/cancel events with timestamps
    """

    def __init__(
        self,
        allowed_event_types: List[str] = DEFAULT_ALLOWED_EVENT_TYPES,
        event_types_description: str = DEFAULT_EVENT_TYPES_CONFIG,
    ):
        """Initialize the EventSpace"""
        super().__init__()
        self._allowed_event_types = allowed_event_types
        self._event_types_description = event_types_description
        self._agent_events: Dict[int, CurrentEvent] = {}
        """Storage for current event of each person: person_id -> CurrentEvent"""

    @property
    def description(self):
        """Description of the environment module for router selection"""
        return f"""Event tracking module for managing person behavior and activities.

**Available Event Commands:**

{self._event_types_description}

**Query and Control:**
- Get current event info: get_current_event(person_id)
  - Returns: event details including type, status, duration, progress percentage, elapsed/remaining time
  
- Stop event: stop_event(person_id, status)
  - status: "completed" or "cancelled"

**Key Features:**
- Each person can maintain one active event at a time
- New events automatically replace previous ones
- Progress tracking based on elapsed vs expected duration
- Timestamps for start, expected end, and actual end times"""

    async def init(self, start_datetime: datetime) -> Any:
        """
        Initialize the EventSpace.

        Args:
            start_datetime: The simulation start time
        """
        await super().init(start_datetime)
        get_logger().info("EventSpace initialized")

    def _create_and_start_event(
        self,
        person_id: int,
        event_type: str,
        event_name: str,
        expected_duration_seconds: float,
    ) -> StartEventResponse:
        """
        Internal helper to create and start an event.

        Args:
            person_id: The ID of the person
            event_type: Type of event
            event_name: Name or description of the event
            expected_duration_seconds: Expected duration in seconds

        Returns:
            Response containing the started event information
        """
        start_time = self.t

        # Check if duration is provided and valid
        if expected_duration_seconds is None or expected_duration_seconds <= 0:
            raise ValueError(
                "expected_duration_seconds is required and must be positive. "
                "Please specify how long the event should take in seconds. "
                "Examples: 3600 for 1 hour, 28800 for 8 hours, 1800 for 30 minutes."
            )

        # Calculate expected end time
        expected_end = start_time + timedelta(seconds=expected_duration_seconds)

        # Create new event
        event = CurrentEvent(
            person_id=person_id,
            event_type=event_type,
            event_name=event_name,
            status="in_progress",
            start_time=start_time,
            expected_end_time=expected_end,
        )

        # Replace any existing event for this person
        if person_id in self._agent_events:
            raise ValueError(f"Event {event_type}/{event_name} already exists for person {person_id}. Please stop the existing event first.")

        self._agent_events[person_id] = event

        get_logger().info(
            f"Event started: person_id={person_id}, event_type={event_type}, event_name={event_name}, duration={expected_duration_seconds}s, end_time={expected_end.isoformat()}"
        )

        return StartEventResponse(
            person_id=person_id,
            event_type=event_type,
            event_name=event_name,
            start_time=start_time,
            expected_end_time=expected_end,
            status="in_progress",
        )

    @tool(readonly=False)
    async def start_event(
        self,
        person_id: int,
        event_type: str,
        event_name: str,
        expected_duration_seconds: float,
    ) -> StartEventResponse:
        """
        Start an event for a person.

        Args:
            person_id: The ID of the person
            event_type: Type of event - one of: sleep, home activity, work, shopping, eating out, leisure and entertainment, other
            event_name: Description or name of the specific activity
            expected_duration_seconds: Expected duration in seconds
                (e.g., 3600 for 1 hour, 28800 for 8 hours, 1800 for 30 minutes)

        Returns:
            Response containing the started event information
        """
        # Validate event_type
        if event_type not in self._allowed_event_types:
            raise ValueError(
                f"Invalid event_type: {event_type}. Must be one of: {', '.join(self._allowed_event_types)}"
            )

        return self._create_and_start_event(
            person_id=person_id,
            event_type=event_type,  # type: ignore
            event_name=event_name,
            expected_duration_seconds=expected_duration_seconds,
        )

    @tool(readonly=True, kind="observe")
    async def get_current_event(self, person_id: int) -> Optional[GetEventResponse]:
        """
        Query current event information for a person.

        Args:
            person_id: The ID of the person

        Returns:
            Response containing event details including type, status, timestamps, duration, and progress, or None if no active event
        """
        if person_id not in self._agent_events:
            return None

        event = self._agent_events[person_id]
        
        # Only return active events, not completed or cancelled ones
        if event.status != "in_progress":
            return None
            
        elapsed = event.get_elapsed_seconds(self.t)
        remaining = event.get_remaining_seconds(self.t)
        progress = event.get_progress_percentage(self.t)

        return GetEventResponse(
            person_id=event.person_id,
            event_type=event.event_type,
            event_name=event.event_name,
            status=event.status,
            start_time=event.start_time,
            expected_end_time=(
                event.expected_end_time if event.expected_end_time else None
            ),
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            progress_percentage=progress,
        )

    @tool(readonly=False)
    async def stop_event(
        self,
        person_id: int,
        status: Literal["completed", "cancelled"] = "completed",
    ) -> EndEventResponse:
        """
        Stop (complete or cancel) the current event for a person.

        Args:
            person_id: The ID of the person
            status: Event end status ("completed" or "cancelled")

        Returns:
            Response indicating success and status message
        """
        if person_id not in self._agent_events:
            return EndEventResponse(
                success=False,
                message=f"No active event for person: {person_id}",
            )

        event = self._agent_events[person_id]
        event.status = status

        get_logger().info(
            f"Event stopped: person_id={person_id}, event_type={event.event_type}, "
            f"event_name={event.event_name}, status={status}, duration={event.get_elapsed_seconds(self.t):.1f}s"
        )

        # Remove the event from active events after stopping
        del self._agent_events[person_id]

        return EndEventResponse(
            success=True,
            message=f"Event {event.event_name} stopped with status: {status}",
        )

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step (update event timestamps).

        Args:
            tick: The number of ticks (1 tick = 1 second)
            t: The current datetime after this step
        """
        self.t = t

    def _dump_state(self) -> dict:
        """
        Dump the internal state of EventSpace for serialization.

        Returns:
            dict: A dictionary containing current events for all persons
        """
        events_data = {}
        for person_id, event in self._agent_events.items():
            event_dict = {
                "person_id": event.person_id,
                "event_type": event.event_type,
                "event_name": event.event_name,
                "status": event.status,
                "start_time": event.start_time,
                "expected_end_time": (
                    event.expected_end_time if event.expected_end_time else None
                ),
            }
            events_data[str(person_id)] = event_dict

        return {"events": events_data}

    def _load_state(self, state: dict):
        """
        Load the internal state of EventSpace from serialized data.

        Args:
            state: The state dictionary produced by _dump_state()
        """
        events_data = state.get("events", {})
        self._agent_events = {}
        for person_id_str, event_dict in events_data.items():
            person_id = int(person_id_str)
            # Convert ISO format strings back to datetime
            event = CurrentEvent(
                person_id=event_dict["person_id"],
                event_type=event_dict.get(
                    "event_type", "other"
                ),  # Default to "other" for backward compatibility
                event_name=event_dict["event_name"],
                status=event_dict["status"],
                start_time=event_dict["start_time"],
                expected_end_time=(
                    event_dict["expected_end_time"]
                    if event_dict.get("expected_end_time")
                    else None
                ),
            )
            self._agent_events[person_id] = event

    async def close(self):
        """
        Close the EventSpace and perform cleanup.
        """
        get_logger().info(
            f"EventSpace closed. Total persons with events: {len(self._agent_events)}"
        )
