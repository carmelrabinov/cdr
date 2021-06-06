import torch
from torch import nn
from typing import List


class MLP(nn.Module):
    """ general MLP Module"""
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [],
                 activation_func: nn.Module = nn.ReLU()):
        super().__init__()
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(activation_func)
            prev_h = h
        model.pop()     # Pop last activation function
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ForwardModel(nn.Module):
    """ Forward model Module to be implemented """
    def __init__(self, z_dim, action_dim, action_hidden_dim=None, z_hidden_dim=None, add_skip_connection=False):
        super().__init__()

        self.add_skip_connection = add_skip_connection
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim
        self.z_hidden_dim = z_hidden_dim

        self.generate_net()

    def generate_net(self) -> nn.Module:
        """ generate model network to be implemented"""
        raise NotImplementedError


class LinearForwardModel(ForwardModel):
    """ simple one linear layer forward model """
    def generate_net(self):
        self.model = nn.Linear(self.z_dim + self.action_dim, self.z_dim, bias=False)

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        z_hat = self.model(x)
        if self.add_skip_connection:
            z_hat = z_hat + z
        return z_hat


class MLPForwardModel(ForwardModel):
    """ 3 layered MLP forward model with relu activations """
    def generate_net(self):
        self.model = MLP(input_size=self.z_dim + self.action_dim,
                         output_size=self.z_dim,
                         hidden_sizes=[self.z_hidden_dim, self.z_hidden_dim],
                         activation_func=nn.ReLU())

    def forward(self, z, a):
        z_hat = torch.cat((z, a), dim=-1)
        z_hat = self.model(z_hat)

        if self.add_skip_connection:
            z_hat += z
        return z_hat


class CFMForwardModel(ForwardModel):
    """ 3 layered MLP that outputs values for a linear transformation matrix to be applied as a forward model """
    def generate_net(self):
        self.model = MLP(input_size=self.z_dim + self.action_dim,
                         output_size=self.z_dim*self.z_dim,
                         hidden_sizes=[self.z_hidden_dim, self.z_hidden_dim],
                         activation_func=nn.ReLU())

    def forward(self, z, a):
        z_hat = torch.cat((z, a), dim=-1)
        Ws = self.model(z_hat).view(z_hat.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
        z_hat = torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1)  # b x z_dim

        if self.add_skip_connection:
            z_hat += z
        return z_hat


class ProjectedForwardModel(ForwardModel):
    """ 3 layered MLP forward model with relu activations, with actions and state projection """

    def generate_net(self):
        self.action_projection = nn.Sequential(nn.Linear(self.action_dim, self.action_hidden_dim), nn.Tanh())
        self.state_projection = nn.Sequential(nn.Linear(self.z_dim, self.z_hidden_dim), nn.Tanh())

        self.model = MLP(input_size=self.z_hidden_dim + self.action_hidden_dim,
                         output_size=self.z_dim,
                         hidden_sizes=[self.z_hidden_dim + self.action_hidden_dim, self.z_hidden_dim],
                         activation_func=nn.ReLU())

    def forward(self, z, action):
        action_projection = self.action_projection(action)
        state_projection = self.state_projection(z)
        z_hat = torch.cat((action_projection, state_projection), dim=-1)
        z_hat = self.model(z_hat)

        if self.add_skip_connection:
            z_hat = z_hat + z
        return z_hat
