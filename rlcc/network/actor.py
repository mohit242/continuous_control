import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=nn.ReLU()):
        """ Network class for PPO actor network

        Args:
            state_dim: dimension of states/observations
            action_dim: dimension of action vector
            hidden_units : list of number of hidden layer neurons
            gate: activation gate
        """
        super().__init__()
        dims = (state_dim, ) + hidden_units + (action_dim, )
        linear_func = lambda a, b: nn.Linear(a, b)
        act_func = lambda a, b: gate
        layers = [f(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:]) for f in (linear_func, act_func)]
        layers = layers[:-1]
        self.network = nn.Sequential(*layers)

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, action=None):
        state = torch.Tensor(state)
        mean = F.tanh(self.network(state))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)

        # No need for entropy as entropy for normal distribution depends on sigma

        return{'a': action,
               'log_prob': log_prob,
               'mean': mean
               }