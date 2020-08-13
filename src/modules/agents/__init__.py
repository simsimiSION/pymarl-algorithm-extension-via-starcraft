REGISTRY = {}

from .rnn_agent import RNNAgent
from .commnet_agent import CommAgent
from .g2a_agent import G2AAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY['commnet'] = CommAgent
REGISTRY['g2a'] = G2AAgent

