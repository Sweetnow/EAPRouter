"""
Self-Reference Effect (SRE) Experiment Environment
Environment for Self-Reference Effect experiment based on V2 framework
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from simframework.env import EnvBase, tool


# Identity types
class IdentityType(str, Enum):
    """Identity types for trait adjectives"""
    SELF = "self"
    FRIEND = "friend"
    OTHER = "other"


# Response models for tool functions
class SubmitEncodingRatingResponse(BaseModel):
    """Response model for submit_encoding_rating() function"""

    agent_id: int = Field(..., description="Agent ID")
    trait: str = Field(..., description="Trait adjective")
    identity: str = Field(..., description="Identity type (self, friend, other)")
    rating: int = Field(..., description="Rating score (1-5)")
    status: str = Field(..., description="Status: 'submitted' or 'encoding_completed'")


class SubmitRecognitionResponse(BaseModel):
    """Response model for submit_recognition() function"""

    agent_id: int = Field(..., description="Agent ID")
    trait: str = Field(..., description="Trait adjective")
    judge_type: str = Field(..., description="Recognition judgment (old/new)")
    is_correct: bool = Field(..., description="Whether the recognition is correct")
    rk_type: Optional[str] = Field(None, description="Remember/Know judgment (remember/know)")
    status: str = Field(..., description="Status: 'submitted' or 'recognition_completed'")


class GetEncodingStatusResponse(BaseModel):
    """Response model for get_encoding_status() function"""

    agent_id: int = Field(..., description="Agent ID")
    completed_traits: List[Dict[str, Any]] = Field(..., description="List of completed encoding ratings")
    remaining_count: int = Field(..., description="Number of remaining traits to rate")


class GetRecognitionStatusResponse(BaseModel):
    """Response model for get_recognition_status() function"""

    agent_id: int = Field(..., description="Agent ID")
    completed_judgments: List[Dict[str, Any]] = Field(..., description="List of completed recognition judgments")
    remaining_count: int = Field(..., description="Number of remaining traits to judge")


class SelfReferenceEffectEnv(EnvBase):
    """Environment for Self-Reference Effect (SRE) experiment based on V2 framework"""

    def __init__(
        self,
        agent_ids: List[int],
        encoding_traits: List[Dict[str, Any]] = None,
        recognition_traits: List[str] = None,
    ):
        """
        Initialize the Self-Reference Effect environment.

        Args:
            agent_ids: List of agent IDs participating in the experiment
            encoding_traits: List of trait dictionaries for encoding phase
                Each dict should have: {"trait": str, "identity": str, "valence": int}
                If None, will use default trait list
            recognition_traits: List of trait adjectives for recognition phase
                If None, will use traits from encoding phase plus new traits
        """
        super().__init__()

        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # Initialize encoding traits if not provided
        if encoding_traits is None:
            # Default trait list - you can customize this
            self.encoding_traits = self._generate_default_encoding_traits()
        else:
            self.encoding_traits = encoding_traits
        
        # Initialize recognition traits if not provided
        if recognition_traits is None:
            # Use encoding traits plus some new traits
            self.recognition_traits = [t["trait"] for t in self.encoding_traits] + self._generate_new_traits()
        else:
            self.recognition_traits = recognition_traits
        
        # Store encoding ratings: agent_id -> list of {trait, identity, rating}
        self._encoding_ratings: Dict[int, List[Dict[str, Any]]] = {
            agent_id: [] for agent_id in agent_ids
        }
        
        # Store recognition judgments: agent_id -> list of {trait, judge_type, is_correct, rk_type}
        self._recognition_judgments: Dict[int, List[Dict[str, Any]]] = {
            agent_id: [] for agent_id in agent_ids
        }
        
        # Track which traits were presented in encoding phase (for correctness checking)
        self._encoding_trait_set = {t["trait"] for t in self.encoding_traits}
        
        self._lock = asyncio.Lock()

    def _generate_default_encoding_traits(self) -> List[Dict[str, Any]]:
        """Generate default trait list for encoding phase"""
        # This is a simplified version - you should load from actual SRE data
        traits = [
            {"trait": "谦和", "identity": "self", "valence": 1},
            {"trait": "善良", "identity": "self", "valence": 1},
            {"trait": "积极", "identity": "friend", "valence": 1},
            {"trait": "整洁", "identity": "other", "valence": 1},
            {"trait": "坦诚", "identity": "other", "valence": 1},
            {"trait": "放荡", "identity": "self", "valence": 2},
            {"trait": "负心", "identity": "other", "valence": 2},
            {"trait": "恶念", "identity": "other", "valence": 2},
            {"trait": "努力", "identity": "other", "valence": 1},
        ]
        return traits

    def _generate_new_traits(self) -> List[str]:
        """Generate new traits for recognition phase (not in encoding)"""
        return ["聪明", "勇敢", "懒惰", "自私"]

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """
        description = f"""{cls.__name__}: Self-Reference Effect experiment environment module.

**Description:** Manages a Self-Reference Effect experiment where agents rate trait adjectives in encoding phase and recognize them in recognition phase.

**Initialization Parameters (excluding llm):**
- agent_ids (List[int]): List of agent IDs participating in the experiment
- encoding_traits (List[Dict], optional): List of trait dictionaries for encoding phase
- recognition_traits (List[str], optional): List of trait adjectives for recognition phase

**Example initialization config:**
```json
{{
  "agent_ids": [101, 102, 103]
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module for router selection and function calling"""
        # Get encoding traits list for display (only for encoding phase)
        encoding_traits_str = "\n".join([
            f"  - {t['trait']} (identity: {t['identity']}, valence: {t.get('valence', 1)})"
            for t in self.encoding_traits
        ])
        
        return f"""You are a Self-Reference Effect (SRE) experiment environment module specialized in managing memory and recognition tasks.

**Experiment Overview:** {self.num_agents} agents are participating in a self-reference effect study.

**Experiment Structure:**
The experiment consists of two phases:
1. **Encoding Phase**: Rate trait adjectives (1-5 scale) associated with different identities (self, friend, other)
2. **Recognition Phase**: Judge whether trait adjectives were presented in the encoding phase (old/new) and provide Remember/Know judgment

**IMPORTANT: YOU MUST ACTIVELY COMPLETE ALL TASKS**

**Encoding Phase Task:**
You MUST complete ratings for ALL {len(self.encoding_traits)} encoding traits. The traits you need to rate are:
{encoding_traits_str}

**STEP-BY-STEP INSTRUCTIONS FOR ENCODING PHASE:**
1. First, call get_encoding_status(agent_id) to see which traits you still need to rate
2. For each remaining trait, call submit_encoding_rating(agent_id, trait, identity, rating)
3. Rating scale (1-5):
   - 1 = Not at all
   - 2 = Slightly
   - 3 = Moderately
   - 4 = Very well
   - 5 = Extremely well
4. Consider your psychological profile when making ratings
5. Continue until all {len(self.encoding_traits)} traits are rated

**Recognition Phase Task:**
You MUST complete judgments for ALL {len(self.recognition_traits)} recognition traits. 

**IMPORTANT FOR RECOGNITION PHASE:**
- You will be presented with trait adjectives one by one
- You need to judge whether EACH trait was presented in the encoding phase (old/new)
- DO NOT assume you know which traits are old or new - rely on your MEMORY of the encoding phase
- Some traits were in encoding phase (old), some were not (new)
- You need to use your memory to distinguish between old and new traits

**STEP-BY-STEP INSTRUCTIONS FOR RECOGNITION PHASE:**
1. First, call get_recognition_status(agent_id) to see which traits you still need to judge
2. For each remaining trait, call submit_recognition(agent_id, trait, judge_type, rk_type)
3. Judge whether each trait was in encoding phase based on YOUR MEMORY:
   - "old" = Yes, you remember it was presented in encoding phase
   - "new" = No, you do not remember it being presented in encoding phase
4. For traits judged as "old", you MUST provide rk_type:
   - "remember" = You remember the specific episode of seeing this trait
   - "know" = You know it was presented but don't remember the specific episode
5. Rely on your memory - do not try to systematically check against a list
6. Consider your memory and self-reference when making judgments
7. Continue until all {len(self.recognition_traits)} traits are judged

**Available Operations (you MUST use these within your plan):**

1. **get_encoding_status(agent_id)**: View your encoding phase progress
   - agent_id: Your agent ID
   - Returns: Completed traits and remaining count
   - **CALL THIS FIRST** to see what you need to do

2. **submit_encoding_rating(agent_id, trait, identity, rating)**: Submit rating for a trait in encoding phase
   - agent_id: Your agent ID (e.g., 101)
   - trait: The trait adjective (e.g., "谦和")
   - identity: Identity type ("self", "friend", or "other")
   - rating: Rating score (1-5, integer)
   - **CALL THIS FOR EACH TRAIT** until all are completed

3. **get_recognition_status(agent_id)**: View your recognition phase progress
   - agent_id: Your agent ID
   - Returns: Completed judgments and remaining count
   - **CALL THIS FIRST** to see what you need to do

4. **submit_recognition(agent_id, trait, judge_type, rk_type=None)**: Submit recognition judgment
   - agent_id: Your agent ID
   - trait: The trait adjective
   - judge_type: "old" or "new"
   - rk_type: "remember" or "know" (REQUIRED if judge_type is "old")
   - **CALL THIS FOR EACH TRAIT** until all are completed

**CRITICAL CONSTRAINTS:**
⚠️ You MUST actively complete ALL tasks - the system will NOT present tasks to you automatically
⚠️ Start by calling get_encoding_status() to see what needs to be done
⚠️ In encoding phase, you MUST rate all {len(self.encoding_traits)} traits
⚠️ Rating must be an integer between 1 and 5
⚠️ In recognition phase, you MUST judge all {len(self.recognition_traits)} traits
⚠️ If judge_type is "old", you MUST provide rk_type ("remember" or "know")
⚠️ Consider your psychological profile and self-reference when making judgments
⚠️ Work systematically through all tasks - don't skip any!
"""

    @tool(readonly=False)
    async def submit_encoding_rating(
        self, agent_id: int, trait: str, identity: str, rating: int
    ) -> SubmitEncodingRatingResponse:
        """
        Submit rating for a trait adjective in encoding phase.

        Args:
            agent_id: The agent's ID
            trait: The trait adjective
            identity: Identity type ("self", "friend", or "other")
            rating: Rating score (1-5, integer)

        Returns:
            Response containing submission status.
        """
        async with self._lock:
            # Validate agent_id
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            # Validate identity
            identity_lower = identity.lower()
            if identity_lower not in ["self", "friend", "other"]:
                raise ValueError(
                    f"Invalid identity: {identity}. Must be one of: self, friend, other"
                )
            
            # Validate rating
            if not isinstance(rating, int):
                rating = int(round(rating))
            rating = max(1, min(5, rating))  # Clamp to 1-5
            
            # Check if this trait-identity combination exists in encoding traits
            trait_found = False
            for encoding_trait in self.encoding_traits:
                if encoding_trait["trait"] == trait and encoding_trait["identity"] == identity_lower:
                    trait_found = True
                    break
            
            if not trait_found:
                raise ValueError(
                    f"Trait '{trait}' with identity '{identity_lower}' not found in encoding traits list"
                )
            
            # Check if already submitted
            for existing in self._encoding_ratings[agent_id]:
                if existing["trait"] == trait and existing["identity"] == identity_lower:
                    raise ValueError(
                        f"Rating for trait '{trait}' with identity '{identity_lower}' has already been submitted. "
                        f"Current rating: {existing['rating']}"
                    )
            
            # Store the rating
            rating_data = {
                "trait": trait,
                "identity": identity_lower,
                "rating": rating,
            }
            self._encoding_ratings[agent_id].append(rating_data)
            
            # Check if all encoding traits are completed
            completed = len(self._encoding_ratings[agent_id])
            all_completed = completed == len(self.encoding_traits)
            
            status = "encoding_completed" if all_completed else "submitted"
            
            # Debug log
            import sys
            print(
                f"[ENV DEBUG] Agent {agent_id} submitted encoding rating: {trait} ({identity_lower}) = {rating} "
                f"({completed}/{len(self.encoding_traits)} completed)",
                file=sys.stderr
            )
            
            return SubmitEncodingRatingResponse(
                agent_id=agent_id,
                trait=trait,
                identity=identity_lower,
                rating=rating,
                status=status,
            )

    @tool(readonly=True)
    async def get_encoding_status(self, agent_id: int) -> GetEncodingStatusResponse:
        """
        Get encoding phase status for a specific agent.

        Args:
            agent_id: The agent's ID

        Returns:
            Response containing completed traits and remaining count.
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            completed_traits = self._encoding_ratings[agent_id].copy()
            remaining_count = len(self.encoding_traits) - len(completed_traits)
            
            return GetEncodingStatusResponse(
                agent_id=agent_id,
                completed_traits=completed_traits,
                remaining_count=remaining_count,
            )

    @tool(readonly=False)
    async def submit_recognition(
        self, agent_id: int, trait: str, judge_type: str, rk_type: Optional[str] = None
    ) -> SubmitRecognitionResponse:
        """
        Submit recognition judgment for a trait adjective.

        Args:
            agent_id: The agent's ID
            trait: The trait adjective
            judge_type: Recognition judgment ("old" or "new")
            rk_type: Remember/Know judgment ("remember" or "know", required if judge_type is "old")

        Returns:
            Response containing judgment status and correctness.
        """
        async with self._lock:
            # Validate agent_id
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            # Validate judge_type
            judge_type_lower = judge_type.lower()
            if judge_type_lower not in ["old", "new"]:
                raise ValueError(
                    f"Invalid judge_type: {judge_type}. Must be 'old' or 'new'"
                )
            
            # Validate rk_type if judge_type is "old"
            if judge_type_lower == "old":
                if rk_type is None:
                    raise ValueError(
                        "rk_type is required when judge_type is 'old'. Must be 'remember' or 'know'"
                    )
                rk_type_lower = rk_type.lower()
                if rk_type_lower not in ["remember", "know"]:
                    raise ValueError(
                        f"Invalid rk_type: {rk_type}. Must be 'remember' or 'know'"
                    )
            else:
                rk_type_lower = None
            
            # Check if trait is in recognition traits
            if trait not in self.recognition_traits:
                raise ValueError(
                    f"Trait '{trait}' not found in recognition traits list"
                )
            
            # Check if already submitted
            for existing in self._recognition_judgments[agent_id]:
                if existing["trait"] == trait:
                    raise ValueError(
                        f"Recognition judgment for trait '{trait}' has already been submitted. "
                        f"Current judgment: {existing['judge_type']}"
                    )
            
            # Determine correctness
            is_correct = False
            if judge_type_lower == "old":
                # Correct if trait was in encoding phase
                is_correct = trait in self._encoding_trait_set
            else:  # judge_type == "new"
                # Correct if trait was NOT in encoding phase
                is_correct = trait not in self._encoding_trait_set
            
            # Store the judgment
            judgment_data = {
                "trait": trait,
                "judge_type": judge_type_lower,
                "is_correct": is_correct,
                "rk_type": rk_type_lower,
            }
            self._recognition_judgments[agent_id].append(judgment_data)
            
            # Check if all recognition traits are completed
            completed = len(self._recognition_judgments[agent_id])
            all_completed = completed == len(self.recognition_traits)
            
            status = "recognition_completed" if all_completed else "submitted"
            
            # Debug log
            import sys
            print(
                f"[ENV DEBUG] Agent {agent_id} submitted recognition: {trait} = {judge_type_lower} "
                f"(correct: {is_correct}, rk: {rk_type_lower}) "
                f"({completed}/{len(self.recognition_traits)} completed)",
                file=sys.stderr
            )
            
            return SubmitRecognitionResponse(
                agent_id=agent_id,
                trait=trait,
                judge_type=judge_type_lower,
                is_correct=is_correct,
                rk_type=rk_type_lower,
                status=status,
            )

    @tool(readonly=True)
    async def get_recognition_status(self, agent_id: int) -> GetRecognitionStatusResponse:
        """
        Get recognition phase status for a specific agent.

        Args:
            agent_id: The agent's ID

        Returns:
            Response containing completed judgments and remaining count.
            Note: Only returns trait names and judge_type to avoid memory contamination.
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(f"Agent ID {agent_id} is not in the experiment")
            
            # Only return minimal information to avoid memory contamination
            # Don't reveal is_correct or rk_type to prevent agents from using feedback
            completed_judgments = [
                {
                    "trait": j.get("trait", ""),
                    "judge_type": j.get("judge_type", ""),
                }
                for j in self._recognition_judgments[agent_id]
            ]
            remaining_count = len(self.recognition_traits) - len(self._recognition_judgments[agent_id])
            
            return GetRecognitionStatusResponse(
                agent_id=agent_id,
                completed_judgments=completed_judgments,
                remaining_count=remaining_count,
            )

    @tool(readonly=True, kind="statistics")
    async def get_all_results(self) -> Dict[str, Any]:
        """
        Get all results for all agents (statistics function).

        Returns:
            Dictionary containing encoding ratings and recognition judgments for all agents.
        """
        async with self._lock:
            return {
                "encoding_ratings": {
                    agent_id: ratings.copy()
                    for agent_id, ratings in self._encoding_ratings.items()
                },
                "recognition_judgments": {
                    agent_id: judgments.copy()
                    for agent_id, judgments in self._recognition_judgments.items()
                },
            }

    async def init(self, start_datetime: datetime):
        """
        Initialize the environment module.
        """
        await super().init(start_datetime)

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        Args:
            tick: The number of ticks (1 tick = 1 second) of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        async with self._lock:
            self.current_datetime = t

    def get_results(self) -> Dict[str, Any]:
        """
        Get all results (synchronous method for result extraction).

        Returns:
            Dictionary containing encoding ratings and recognition judgments for all agents.
        """
        return {
            "encoding_ratings": {
                agent_id: [rating.copy() for rating in ratings]
                for agent_id, ratings in self._encoding_ratings.items()
            },
            "recognition_judgments": {
                agent_id: [judgment.copy() for judgment in judgments]
                for agent_id, judgments in self._recognition_judgments.items()
            },
        }

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "agent_ids": self.agent_ids,
            "num_agents": self.num_agents,
            "encoding_traits": self.encoding_traits,
            "recognition_traits": self.recognition_traits,
            "encoding_ratings": self._encoding_ratings,
            "recognition_judgments": self._recognition_judgments,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.agent_ids = state.get("agent_ids", [])
        self.num_agents = state.get("num_agents", 0)
        self.encoding_traits = state.get("encoding_traits", [])
        self.recognition_traits = state.get("recognition_traits", [])
        self._encoding_ratings = state.get("encoding_ratings", {})
        self._recognition_judgments = state.get("recognition_judgments", {})


__all__ = ["SelfReferenceEffectEnv", "IdentityType"]

