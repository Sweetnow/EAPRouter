"""
Implicit Association Test (IAT) Experiment Environment
Environment for Implicit Association Test experiment based on V2 framework
"""
import asyncio
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from eaprouter.env import EnvBase, tool


# Response models for tool functions
class TrialInfo(BaseModel):
    """Response model for get_next_trial() function"""

    trial_id: int = Field(..., description="Trial ID (sequential number)")
    block_code: str = Field(..., description="Block code (identity_practice, valence_practice, congruent, identity_switch, incongruent)")
    stimuli: str = Field(..., description="Stimulus word (in Chinese)")
    identity: Optional[str] = Field(None, description="Identity category (1=self, 2=others, or None if not applicable)")
    valence: Optional[str] = Field(None, description="Valence category (1=positive, 2=negative, or None if not applicable)")
    left_label: str = Field(..., description="Left label")
    right_label: str = Field(..., description="Right label")
    correct_key: str = Field(..., description="Correct response key (z or m)")
    instruction: str = Field(..., description="Instruction for this trial")


class SubmitTrialResponse(BaseModel):
    """Response model for submit_trial_response() function"""

    agent_id: int = Field(..., description="Agent ID")
    trial_id: int = Field(..., description="Trial ID")
    key_press: str = Field(..., description="Pressed key (z or m)")
    rt: float = Field(..., description="Response time in seconds")
    corr: int = Field(..., description="Correctness (1=correct, 0=incorrect)")
    status: str = Field(..., description="Status: 'submitted' or 'completed'")


class ImplicitAssociationTestEnv(EnvBase):
    """Environment for Implicit Association Test (IAT) experiment based on V2 framework"""

    # Standard IAT trial sequence
    # This is a simplified version - in practice, trials should be loaded from data files
    # or generated according to IAT protocol
    STANDARD_TRIALS = [
        # Block 1: Identity Practice (12 trials)
        {"block_code": "identity_practice", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "identity_practice", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "identity_practice", "stimuli": "本人", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "他", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "identity_practice", "stimuli": "我的", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "别人", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "identity_practice", "stimuli": "自个", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "她", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "identity_practice", "stimuli": "俺", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "identity_practice", "stimuli": "他的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        
        # Block 2: Valence Practice (12 trials)
        {"block_code": "valence_practice", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "valence_practice", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "valence_practice", "stimuli": "友好", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "冷漠", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "valence_practice", "stimuli": "诚实", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "自私", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "valence_practice", "stimuli": "慷慨", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "卑鄙", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "valence_practice", "stimuli": "真诚", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "valence_practice", "stimuli": "狡猾", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        
        # Block 3: Congruent (self+positive, others+negative) - 48 trials
        # Mix of identity and valence words
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "others", "right_label": "self", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "negative", "right_label": "positive", "correct_key": "m"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        {"block_code": "congruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "negative", "right_label": "positive", "correct_key": "z"},
        
        # Block 4: Identity Switch (12 trials)
        {"block_code": "identity_switch", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "identity_switch", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "identity_switch", "stimuli": "本人", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "他", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "identity_switch", "stimuli": "我的", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "别人", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "identity_switch", "stimuli": "自个", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "她", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "identity_switch", "stimuli": "俺", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "identity_switch", "stimuli": "他的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        
        # Block 5: Incongruent (self+negative, others+positive) - 48 trials
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "他们", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "虚伪", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "可靠", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "自我", "identity": "1", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "她的", "identity": "2", "valence": None, "left_label": "self", "right_label": "others", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "吝啬", "identity": None, "valence": "2", "left_label": "positive", "right_label": "negative", "correct_key": "m"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
        {"block_code": "incongruent", "stimuli": "善良", "identity": None, "valence": "1", "left_label": "positive", "right_label": "negative", "correct_key": "z"},
    ]

    def __init__(self, agent_ids: List[int], trials: Optional[List[Dict]] = None):
        """
        Initialize the Implicit Association Test environment.

        Args:
            agent_ids: List of agent IDs participating in the experiment
            trials: Optional custom trial sequence. If None, uses STANDARD_TRIALS
        """
        super().__init__()

        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)

        # Use custom trials or standard trials
        self.trials = trials if trials is not None else self.STANDARD_TRIALS
        self.total_trials = len(self.trials)

        # Track progress for each agent: {agent_id: current_trial_index}
        self._trial_progress: Dict[int, int] = {
            agent_id: 0 for agent_id in agent_ids
        }

        # Store responses: {agent_id: [{trial_id, key_press, rt, corr, ...}]}
        self._responses: Dict[int, List[Dict]] = {
            agent_id: [] for agent_id in agent_ids
        }

        self._lock = asyncio.Lock()

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """
        description = f"""{cls.__name__}: Implicit Association Test (IAT) experiment environment module.

**Description:** Manages an IAT experiment where agents perform implicit association tests to measure implicit self-esteem through reaction times and accuracy.

**Initialization Parameters (excluding llm):**
- agent_ids (List[int]): List of agent IDs participating in the experiment
- trials (Optional[List[Dict]]): Optional custom trial sequence

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
        return f"""You are an Implicit Association Test (IAT) experiment environment module specialized in measuring implicit self-esteem.

**Experiment Overview:** {self.num_agents} agents are participating in an IAT experiment to measure implicit associations between self/others and positive/negative concepts.

**Experiment Structure:**
The IAT consists of 5 blocks with a total of {self.total_trials} trials:
1. **Identity Practice** (12 trials): Practice categorizing identity words (self vs. others)
2. **Valence Practice** (12 trials): Practice categorizing valence words (positive vs. negative)
3. **Congruent Block** (48 trials): Self + Positive, Others + Negative (consistent associations)
4. **Identity Switch** (12 trials): Practice with reversed identity labels
5. **Incongruent Block** (48 trials): Self + Negative, Others + Positive (inconsistent associations)

**Task Instructions:**
- You will see a word (stimulus) in the center of the screen
- Two category labels are shown at the top (left and right)
- You must press the correct key (z for left, m for right) as quickly and accurately as possible
- Respond based on which category the word belongs to
- Try to respond as fast as you can while maintaining accuracy

**Available Operations (you MUST use these within your plan):**
1. **get_next_trial(agent_id)**: Get the next trial information
   - agent_id: Your agent ID (e.g., 101, 102)
   - Returns: Trial information including stimulus word, labels, correct key, and instruction
   - You should call this to get each trial before responding

2. **submit_trial_response(agent_id, trial_id, key_press, rt)**: Submit your response for a trial
   - agent_id: Your agent ID
   - trial_id: The trial ID from get_next_trial()
   - key_press: The key you pressed ("z" or "m")
   - rt: Your response time in seconds (should be a realistic reaction time, typically 0.3-2.0 seconds)
   - The environment will automatically check correctness and record your response

**CRITICAL CONSTRAINT:**
⚠️ You MUST complete all {self.total_trials} trials in order.
⚠️ For each trial, you MUST:
   1. First call get_next_trial() to get the trial information
   2. Then call submit_trial_response() with your response
⚠️ Response times should be realistic (typically 0.3-2.0 seconds for correct responses, may be longer for incorrect ones)
⚠️ Consider your psychological profile when making associations - your implicit self-esteem may influence your reaction times
⚠️ In congruent blocks (self+positive), you should respond faster if you have high self-esteem
⚠️ In incongruent blocks (self+negative), you may respond slower if you have high self-esteem
"""

    def _get_instruction(self, block_code: str, identity: Optional[str], valence: Optional[str]) -> str:
        """Generate instruction text for a trial"""
        if block_code == "identity_practice":
            return "Categorize the word as 'self' (press m) or 'others' (press z)"
        elif block_code == "valence_practice":
            return "Categorize the word as 'positive' (press m) or 'negative' (press z)"
        elif block_code == "congruent":
            if identity:
                return "Categorize the identity word: 'self' (press m) or 'others' (press z)"
            else:
                return "Categorize the valence word: 'positive' (press m) or 'negative' (press z)"
        elif block_code == "identity_switch":
            return "Categorize the word as 'self' (press z) or 'others' (press m) - NOTE: labels are reversed!"
        elif block_code == "incongruent":
            if identity:
                return "Categorize the identity word: 'self' (press z) or 'others' (press m) - NOTE: labels are reversed!"
            else:
                return "Categorize the valence word: 'positive' (press z) or 'negative' (press m) - NOTE: labels are reversed!"
        return "Categorize the word according to the labels"

    @tool(readonly=True, kind="observe")
    async def get_next_trial(self, agent_id: int) -> TrialInfo:
        """
        Get the next trial information for an agent.

        Args:
            agent_id: The agent's ID

        Returns:
            TrialInfo containing trial details
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(
                    f"Agent ID {agent_id} is not in the experiment. Valid IDs: {self.agent_ids}"
                )

            current_index = self._trial_progress[agent_id]

            if current_index >= self.total_trials:
                raise ValueError(
                    f"Agent {agent_id} has completed all {self.total_trials} trials. No more trials available."
                )

            trial_data = self.trials[current_index]
            instruction = self._get_instruction(
                trial_data["block_code"],
                trial_data.get("identity"),
                trial_data.get("valence")
            )

            return TrialInfo(
                trial_id=current_index + 1,  # 1-indexed
                block_code=trial_data["block_code"],
                stimuli=trial_data["stimuli"],
                identity=trial_data.get("identity"),
                valence=trial_data.get("valence"),
                left_label=trial_data["left_label"],
                right_label=trial_data["right_label"],
                correct_key=trial_data["correct_key"],
                instruction=instruction,
            )

    @tool(readonly=False)
    async def submit_trial_response(
        self, agent_id: int, trial_id: int, key_press: str, rt: float
    ) -> SubmitTrialResponse:
        """
        Submit response for a trial.

        Args:
            agent_id: The agent's ID
            trial_id: The trial ID (from get_next_trial)
            key_press: The pressed key ("z" or "m")
            rt: Response time in seconds

        Returns:
            SubmitTrialResponse containing correctness and status
        """
        async with self._lock:
            # Validate agent_id
            if agent_id not in self.agent_ids:
                raise ValueError(
                    f"Agent ID {agent_id} is not in the experiment. Valid IDs: {self.agent_ids}"
                )

            # Validate trial_id
            expected_index = trial_id - 1  # Convert to 0-indexed
            if expected_index != self._trial_progress[agent_id]:
                raise ValueError(
                    f"Trial ID mismatch. Expected trial {self._trial_progress[agent_id] + 1}, got {trial_id}. "
                    f"Please call get_next_trial() first."
                )

            if expected_index >= self.total_trials:
                raise ValueError(f"Trial ID {trial_id} is out of range. Total trials: {self.total_trials}")

            # Validate key_press
            key_press = key_press.lower().strip()
            if key_press not in ["z", "m"]:
                raise ValueError(f"Invalid key_press '{key_press}'. Must be 'z' or 'm'")

            # Validate rt (should be positive and reasonable)
            if rt < 0:
                rt = 0.0
            if rt > 10.0:  # Cap at 10 seconds
                rt = 10.0

            # Get trial data
            trial_data = self.trials[expected_index]
            correct_key = trial_data["correct_key"].lower()

            # Check correctness
            corr = 1 if key_press == correct_key else 0

            # Store response
            response_data = {
                "trial_id": trial_id,
                "block_code": trial_data["block_code"],
                "stimuli": trial_data["stimuli"],
                "identity": trial_data.get("identity"),
                "valence": trial_data.get("valence"),
                "left_label": trial_data["left_label"],
                "right_label": trial_data["right_label"],
                "correct_key": correct_key,
                "key_press": key_press,
                "rt": rt,
                "corr": corr,
            }
            self._responses[agent_id].append(response_data)

            # Update progress
            self._trial_progress[agent_id] += 1

            # Check if completed
            all_completed = self._trial_progress[agent_id] >= self.total_trials
            status = "completed" if all_completed else "submitted"

            # Debug log
            import sys
            print(
                f"[ENV DEBUG] Agent {agent_id} trial {trial_id}: key={key_press}, rt={rt:.3f}s, corr={corr} "
                f"({self._trial_progress[agent_id]}/{self.total_trials} completed)",
                file=sys.stderr
            )

            return SubmitTrialResponse(
                agent_id=agent_id,
                trial_id=trial_id,
                key_press=key_press,
                rt=rt,
                corr=corr,
                status=status,
            )

    @tool(readonly=True, kind="observe")
    async def get_my_progress(self, agent_id: int) -> Dict:
        """
        Get progress for a specific agent.

        Args:
            agent_id: The agent's ID

        Returns:
            Dictionary containing progress information
        """
        async with self._lock:
            if agent_id not in self.agent_ids:
                raise ValueError(
                    f"Agent ID {agent_id} is not in the experiment. Valid IDs: {self.agent_ids}"
                )

            completed = self._trial_progress[agent_id]
            responses = self._responses[agent_id]

            # Calculate accuracy
            if responses:
                correct_count = sum(1 for r in responses if r["corr"] == 1)
                accuracy = correct_count / len(responses)
                avg_rt = sum(r["rt"] for r in responses) / len(responses)
            else:
                accuracy = 0.0
                avg_rt = 0.0

            return {
                "agent_id": agent_id,
                "completed_trials": completed,
                "total_trials": self.total_trials,
                "progress_percent": (completed / self.total_trials * 100) if self.total_trials > 0 else 0,
                "accuracy": accuracy,
                "average_rt": avg_rt,
            }

    @tool(readonly=True, kind="statistic")
    async def get_all_progress(self) -> Dict[int, Dict]:
        """
        Get progress for all agents.

        Returns:
            Dictionary mapping agent_id to progress information
        """
        async with self._lock:
            result = {}
            for agent_id in self.agent_ids:
                # Use the same logic as get_my_progress
                completed = self._trial_progress[agent_id]
                responses = self._responses[agent_id]

                if responses:
                    correct_count = sum(1 for r in responses if r["corr"] == 1)
                    accuracy = correct_count / len(responses)
                    avg_rt = sum(r["rt"] for r in responses) / len(responses)
                else:
                    accuracy = 0.0
                    avg_rt = 0.0

                result[agent_id] = {
                    "completed_trials": completed,
                    "total_trials": self.total_trials,
                    "progress_percent": (completed / self.total_trials * 100) if self.total_trials > 0 else 0,
                    "accuracy": accuracy,
                    "average_rt": avg_rt,
                }
            return result

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

    def get_results(self) -> Dict[int, List[Dict]]:
        """
        Get all trial responses (synchronous method for result extraction).

        Returns:
            Dictionary mapping agent_id to list of response dictionaries
        """
        return {
            agent_id: [response.copy() for response in responses]
            for agent_id, responses in self._responses.items()
        }

    def _dump_state(self) -> dict:
        """Serialize state"""
        return {
            "agent_ids": self.agent_ids,
            "num_agents": self.num_agents,
            "total_trials": self.total_trials,
            "trial_progress": self._trial_progress,
            "responses": self._responses,
        }

    def _load_state(self, state: dict):
        """Deserialize state"""
        self.agent_ids = state.get("agent_ids", [])
        self.num_agents = state.get("num_agents", len(self.agent_ids))
        self.total_trials = state.get("total_trials", 0)
        self._trial_progress = state.get("trial_progress", {})
        self._responses = state.get("responses", {})


__all__ = ["ImplicitAssociationTestEnv", "TrialInfo", "SubmitTrialResponse"]

