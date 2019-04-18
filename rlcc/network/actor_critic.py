import torch
import torch.nn as nn
import torch.functional as F
from .actor import PPOActor
from .critic import PPOCritic


class PPOActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=nn.ReLU()):
        """ Network class for PPO actor network

        Args:
            state_dim: dimension of states/observations
            action_dim: dimension of action vector
            hidden_units: list of number of hidden layer neurons
            gate: activation gate
        """
        super().__init__()
        self.actor = PPOActor(state_dim, action_dim, hidden_units, gate)
        self.critic = PPOCritic(state_dim, hidden_units, gate)

    def forward(self, state, action=None):
        y = self.actor(state, action)
        y.update(self.critic(state))
        return y

