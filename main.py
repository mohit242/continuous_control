from rlcc.agent import PPOAgent
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import argparse
import json
import pickle
import torch as torch


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", help="File path of config json file", type=str, default="config.json")
    argparser.add_argument("--play", help="sets mode to play instead of train", action="store_true")
    print("Loading params from config.json .....")
    args = argparser.parse_args()
    with open(args.config, 'r') as f:
        params = json.load(f)
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86", no_graphics=params['no_graphics'])
    agent = PPOAgent(env, steps_per_epoch=params['steps_per_epoch'], gradient_clip=params['gradient_clip'],
                     gamma=params['gamma'], clip_ratio=params['clip_ratio'], device=params['device'],
                     minibatch_size=params['minibatch_size'])
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    scores = []
    scores_window = deque(maxlen=100)
    i = 1
    if not args.play:
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
    else:
        agent.actor_critic.load_state_dict(torch.load('ppo.pth'))
        score = agent.eval_step()
        print("Score : {}".format(score))