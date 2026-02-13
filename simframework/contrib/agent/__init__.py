"""
Agent implementations for SimFramework v2.

Available agents:
"""

from simframework.contrib.agent.commons_tragedy_agent import CommonsTragedyAgent
from simframework.contrib.agent.public_goods_agent import PublicGoodsAgent
from simframework.contrib.agent.prisoners_dilemma_agent import PrisonersDilemmaAgent
from simframework.contrib.agent.trust_game_agent import TrustGameAgent
from simframework.contrib.agent.volunteer_dilemma_agent import VolunteerDilemmaAgent

__all__ = [
    "CommonsTragedyAgent",
    "PublicGoodsAgent",
    "PrisonersDilemmaAgent",
    "TrustGameAgent",
    "VolunteerDilemmaAgent",
]
