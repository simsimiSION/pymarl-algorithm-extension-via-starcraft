REGISTRY = {}

from .rnn_agent import RNNAgent
from .commnet_agent import CommAgent
from .g2a_agent import G2AAgent
from .maven_agent import MAVENAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY['commnet'] = CommAgent
REGISTRY['g2a'] = G2AAgent
REGISTRY['maven'] = MAVENAgent

