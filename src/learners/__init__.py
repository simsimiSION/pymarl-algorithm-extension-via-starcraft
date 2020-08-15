from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qatten_learner import QattenLearner
from .maven_learner import MAVENLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["maven_learner"] = MAVENLearner

