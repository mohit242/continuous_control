import torch
import numpy as np
from .network.actor_critic import PPOActorCritic
import sys

class PPOAgent:

    def __init__(self, env, steps_per_epoch=20, gradient_clip=1, gamma=0.995, clip_ratio=0.2, device='cpu',
                 minibatch_size=200):
        super().__init__()
        self.device = device
        self.minibatch_size = minibatch_size
        self.env = env
        self.gradient_clip = gradient_clip
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.epsilon = clip_ratio
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_dim = self.brain.vector_action_space_size
        self.state_dim = self.brain.vector_observation_space_size

        self.actor_critic = PPOActorCritic(self.state_dim, self.action_dim)
        self.actor_critic.to(device)

        self.opt = torch.optim.Adam(self.actor_critic.parameters())


    def collect_trajectories(self):
        history = {'states': [], 'rewards': [], 'actions': [], 'log_prob': [], 'values': []}

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        while True:
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
            dones = env_info.local_done
            if np.any(dones):
                break

        history['actions'] = torch.stack(history['actions']).detach()
        history['log_prob'] = torch.stack(history['log_prob']).detach()
        history['states'] = torch.stack(history['states']).detach()
        history['values'] = torch.stack(history['values'])
        return history

    def surrogate_func(self, trajectory):
        ind = np.random.choice(np.arange(1001), (self.minibatch_size, ))
        discount = self.gamma ** np.arange(len(trajectory['rewards']))
        rewards = np.asarray(trajectory['rewards']) * discount[:, np.newaxis]
        rewards_futures = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_futures, axis=1)
        std = np.std(rewards_futures, axis=1) + 1.0e-10

        rewards_normalized = (rewards_futures - mean[:, np.newaxis]) / std[:, np.newaxis]
        # using mean and std of whole trajectory
        # rewards_normalized = (rewards_futures - np.mean(rewards_futures)) / (np.std(rewards_futures) + 1.0e-10)

        actions = trajectory['actions'][ind]
        old_probs = trajectory['log_prob'][ind]
        rewards = torch.from_numpy(rewards_normalized[ind]).to(self.device)
        advantage = rewards.unsqueeze(-1).float() - trajectory['values'][ind].detach()
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1.0e-10)

        output = self.actor_critic(trajectory['states'][ind], actions)
        new_probs = output['log_prob']
        v_loss = torch.pow(rewards.unsqueeze(-1).float() - output['v'], 2)

        ratio = (new_probs - old_probs).exp()
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon).float()
        clipped_surrogate = torch.min(ratio*advantage, clip*advantage)
        return torch.mean(-clipped_surrogate) + 0.5*torch.mean(v_loss)

    def learn_step(self):

        trajectory = self.collect_trajectories()
        for _ in range(self.steps_per_epoch):
            clipped_surrogate = self.surrogate_func(trajectory)
            self.opt.zero_grad()
            clipped_surrogate.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.gradient_clip)
            self.opt.step()

        score = np.array(trajectory['rewards']).sum() / self.num_agents
        return score

    def eval_step(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        rewards_history = []
        while True:
            states = torch.Tensor(states).to(self.device)
            output = self.actor_critic(states)
            actions = output['a']
            env_info = self.env.step(actions.numpy())[self.brain_name]
            next_states = env_info.vector_observations
            rewards_history.append(env_info.rewards)
            states = next_states
            dones = env_info.local_done
            if np.any(dones):
                break
        score = np.sum(rewards_history) / self.num_agents
        return score


