REGISTRY = {}

from .rnn_agent import RNNAgent
from .commnet_agent import CommAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY['commnet'] = CommAgent

