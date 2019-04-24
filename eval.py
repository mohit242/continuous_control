from rlcc.agent import PPOAgent
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import pickle
import torch as torch


if __name__=="__main__":
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86")
    agent = PPOAgent(env)
    agent.actor_critic.load_state_dict(torch.load('ppo.pth'))
    for _ in range(10):
        score = agent.eval_step()
        print("Score this session -  {:+8f}".format(score))
