from rlcc.agent import PPOAgent
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import pickle
import torch as torch


if __name__=="__main__":
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86")
    agent = PPOAgent(env)
    torch.manual_seed(0)
    np.random.seed(0)
    scores = []
    scores_window = deque(maxlen=100)
    i = 1
    while True:
        score = agent.learn_step()
        scores.append(score)
        scores_window.append(score)
        print("\rEpisode- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)), end="")
        if i%10 == 0:
            print("\rEpisode- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)))
        if np.mean(scores_window) >= 30:
            break
        if i == 2000:
            break
        i += 1

    with open("ppo.pk", "wb") as f:
        pickle.dump(scores, f)

    torch.save(agent.actor_critic.state_dict(), 'ppo.pth')
