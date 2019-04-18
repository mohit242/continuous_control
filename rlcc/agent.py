import torch
import numpy as np
from .network.actor_critic import PPOActorCritic
import sys

class PPOAgent:

    def __init__(self, env, steps_per_epoch=100, gamma=0.995, epsilon=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.env = env
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.epsilon = epsilon
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_dim = self.brain.vector_action_space_size
        self.state_dim = self.brain.vector_observation_space_size

        self.actor_critic = PPOActorCritic(self.state_dim, self.action_dim)
        self.actor_critic.to(device)

        self.opt = torch.optim.Adam(self.actor_critic.parameters())

    def preprocess_actions(self, actions):
        pass

    def collect_trajectories(self):
        history = {'states': [], 'rewards': [], 'actions': [], 'log_prob': [], 'values': []}

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        for _ in range(self.steps_per_epoch):
            states = torch.Tensor(states).to(self.device)
            output = self.actor_critic(states)
            actions = output['a']
            history['actions'].append(actions)
            history['states'].append(states)
            history['log_prob'].append(output['log_prob'])
            history['values'].append(output['v'])
            env_info = self.env.step(actions.numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            history['rewards'].append(rewards)
            states = next_states

        history['actions'] = torch.stack(history['actions']).detach()
        history['log_prob'] = torch.stack(history['log_prob']).detach()
        history['states'] = torch.stack(history['states']).detach()
        history['values'] = torch.stack(history['values'])
        return history

    def surrogate_func(self, trajectory):

        discount = self.gamma ** np.arange(len(trajectory['rewards']))
        rewards = np.asarray(trajectory['rewards']) * discount[:, np.newaxis]
        rewards_futures = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_futures, axis=1)
        std = np.std(rewards_futures, axis=1) + 1.0e-10

        rewards_normalized = (rewards_futures - mean[:, np.newaxis]) / std[:, np.newaxis]

        actions = trajectory['actions']
        old_probs = trajectory['log_prob']
        rewards = torch.from_numpy(rewards_normalized).to(self.device)
        advantage = rewards.unsqueeze(-1).float() - trajectory['values'].detach()

        output = self.actor_critic(trajectory['states'], actions)
        new_probs = output['log_prob']
        v_loss = torch.pow(rewards.unsqueeze(-1).float() - output['v'], 2)

        ratio = new_probs/old_probs
        sys.stdout.flush()
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon).float()
        clipped_surrogate = torch.min(ratio*advantage, clip*advantage)
        return torch.mean(clipped_surrogate + 0.5*v_loss)

    def learn_step(self):

        trajectory = self.collect_trajectories()
        # TODO: add as parameter
        for _ in range(10):
            clipped_surrogate = self.surrogate_func(trajectory)
            self.opt.zero_grad()
            clipped_surrogate.backward()
            self.opt.step()

        score = np.array(trajectory['rewards']).sum() / self.num_agents
        return score

    def eval_step(self):
        pass
