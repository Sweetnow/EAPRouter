"""
Agent implementations for EAPRouter v2.

Available agents:
"""

from eaprouter.contrib.agent.commons_tragedy_agent import CommonsTragedyAgent
from eaprouter.contrib.agent.public_goods_agent import PublicGoodsAgent
from eaprouter.contrib.agent.prisoners_dilemma_agent import PrisonersDilemmaAgent
from eaprouter.contrib.agent.trust_game_agent import TrustGameAgent
from eaprouter.contrib.agent.volunteer_dilemma_agent import VolunteerDilemmaAgent

__all__ = [
    "CommonsTragedyAgent",
    "PublicGoodsAgent",
    "PrisonersDilemmaAgent",
    "TrustGameAgent",
    "VolunteerDilemmaAgent",
]
