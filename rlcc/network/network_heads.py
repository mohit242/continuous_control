#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_body=None,
                 critic_body=None
                 ):
        super().__init__()
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = nn.Linear(self.actor_body.feature_dim, action_dim)
        self.fc_critic = nn.Linear(self.critic_body.feature_dim, 1)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())

        self.std = nn.Parameter(torch.zeros(action_dim))

        self.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, action=None):
        state = torch.Tensor(state)
        phi_a = self.actor_body(state)
        phi_v = self.critic_body(state)
        mean = F.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v
                }

