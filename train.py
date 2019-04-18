from rlcc.agent import PPOAgent
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import pickle
if __name__=="__main__":
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86")
    agent = PPOAgent(env, steps_per_epoch=10, epsilon=0.2)

    scores = []
    scores_window = deque(maxlen=100)
    i = 0
    while True:
        score = agent.learn_step()
        scores.append(score)
        scores_window.append(score)
        print("\rEpoch- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)), end="")
        if i%10 == 0:
            print("\rEpoch- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)))
        if np.mean(scores_window) >= 30:
            break
        i +=1
        if i == 2000:
            break

    with open("scores.pk", "wb") as f:
        pickle.dump(scores, f)
