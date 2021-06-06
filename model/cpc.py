from typing import Callable

import torch
from torch import nn, Tensor

from model.encoder import Resnet18Encoder, ConvNetFMEncoder, LiteConvNetEncoder
from model.forward_model import LinearForwardModel, CFMForwardModel, MLPForwardModel, ProjectedForwardModel


class ControlCPC(nn.Module):
    def __init__(self, config):
        super(ControlCPC, self).__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = [1, 3, 10]
        self.batch_size = config["batch_size"]
        self.n_neg_actions = config["n_neg_actions"] if "n_neg_actions" in config else 100

        # Architecture
        self.z_dim = config["z_dim"]
        self.z_skip_connection = config["z_delta"]
        self.action_dim = config["action_dim"]
        self.image_size = config["image_size"] if "image_size" in config else 128
        self.observation_dim = [self.image_size, self.image_size, 3]
        self.action_hidden_dim = config["action_hidden_dim"]
        self.z_hidden_dim = config["z_hidden_dim"]

        # loss function
        self.temperature = config["temperature"]
        self.similarity_func_name = config["similarity_function"]
        if self.similarity_func_name == "dot_product_similarity":
            self.similarity_func = self.dot_product_similarity
        elif self.similarity_func_name == "mse_similarity":
            self.similarity_func = self.mse_similarity
        elif self.similarity_func_name == "cosine_similarity":
            self.similarity_func = self.cosine_similarity
        else:
            assert False, f"Error! can't recognize similarity function: {config['similarity_function']}"

        # init encoder, default option is ResNet-18
        if "encoder_type" in config and config["encoder_type"] == "ConvNet":
            self.encoder = ConvNetFMEncoder(output_dim=self.z_dim, input_dim=self.observation_dim)
        elif "encoder_type" in config and config["encoder_type"] == "LiteConvNet":
            self.encoder = LiteConvNetEncoder(output_dim=self.z_dim, input_dim=self.observation_dim)
        elif config["use_oracle_states"]:
            self.encoder = nn.Identity()
        else:
            self.encoder = Resnet18Encoder(input_dim=self.observation_dim, output_dim=self.z_dim)

        # init forward model
        fm_name = config["forward_model_name"]
        if fm_name == "linear":
            self.fm_func = LinearForwardModel
        elif fm_name == "MLP":
            self.fm_func = MLPForwardModel
        elif fm_name == "CFM":
            self.fm_func = CFMForwardModel
        elif fm_name == "projected":
            self.fm_func = ProjectedForwardModel
        else:
            assert False, f"Error! can't recognize forward model type: {fm_name}"
        self.forward_model = self.fm_func(z_dim=self.z_dim,
                                          z_hidden_dim=self.z_hidden_dim,
                                          action_dim=self.action_dim,
                                          action_hidden_dim=self.action_hidden_dim,
                                          add_skip_connection=self.z_skip_connection)

    def compute_fm_accuracy(self, z: Tensor, z_next: Tensor, z_next_hat: Tensor, actions: Tensor,
                            similarity_func: Callable[[Tensor, Tensor], Tensor] = None):
        """
        calculate contrastive accuracy (correctly predicting the positive sample)
        where all negative samples are generated by changing the action input to the forward model b a random action
        this function test the quality of the forward model
        :param z: representation vectors at time t
        :param z_next: representation vectors at time t+1
        :param z_next_hat: predicted representation vectors at time t+1
        :param actions: actions taked at time t
        :param similarity_func: similarity function
        :return: top k accuracy measurements
        """
        assert z_next.size() == z_next_hat.size(), "Error, z_next and z_next_hat must have the same shape"

        batch_size = z_next.shape[0]
        if similarity_func is None:
            similarity_func = self.similarity_func

        # Calculate positive similarities
        similarity_mat = similarity_func(z_next_hat, z_next)[torch.arange(batch_size), torch.arange(batch_size)].unsqueeze(1)

        # Add negative samples from negative actions
        for i in range(1, min(self.n_neg_actions, batch_size)):
            neg_actions = torch.cat((actions[i:], actions[:i]))
            z_next_hat_wrong_action = self.forward_model(z, neg_actions)
            z_next_hat_wrong_action_dist = similarity_func(z_next_hat_wrong_action, z_next)[torch.arange(batch_size), torch.arange(batch_size)].unsqueeze(1)
            similarity_mat = torch.cat((similarity_mat, z_next_hat_wrong_action_dist), dim=1)

        # Compute accuracy
        accuracy = {}
        for k in self.top_k:
            _, preds = similarity_mat.topk(k=k, dim=1)
            labels = torch.zeros_like(preds)
            accuracy[k] = preds.eq(labels).float().mean() * k

        return accuracy

    def compute_loss(self, z: Tensor, z_next: Tensor, z_next_hat: Tensor, actions: Tensor,
                     similarity_func: Callable[[Tensor, Tensor], Tensor] = None):
        """
        generates similarity matrix by defining true and contrastive examples and run the INCE loss function
        :param z: representation vectors at time t
        :param z_next: representation vectors at time t+1
        :param z_next_hat: predicted representation vectors at time t+1
        :param actions: actions taked at time t
        :param similarity_func: similarity function
        :return: loss and contrastive accuracy
        """
        assert z_next.size() == z_next_hat.size(), "Error, z_next and z_next_hat must have the same shape"

        batch_size = z_next.shape[0]
        if similarity_func is None:
            similarity_func = self.similarity_func

        # calculate similarity matrix
        similarity_mat = similarity_func(z_next, z_next_hat)

        # add negative examples from negative actions, meaning, use positive observation and next_observation,
        # and replace the action to generate new contrastive samples
        if self.n_neg_actions > 0:
            for i in range(1, min(self.n_neg_actions, batch_size)):
                neg_actions = torch.cat((actions[i:], actions[:i]))
                z_next_hat_wrong_action = self.forward_model(z, neg_actions)
                z_next_hat_wrong_action_dist = similarity_func(z_next_hat_wrong_action, z_next)[torch.arange(batch_size), torch.arange(batch_size)].unsqueeze(1)
                similarity_mat = torch.cat((similarity_mat, z_next_hat_wrong_action_dist), dim=1)

        return self.ince_loss(similarity_matrix=similarity_mat, positive_on_diagonal=True)

    def ince_loss(self, similarity_matrix: Tensor, positive_on_diagonal: bool = True):
        """
        compute INCE loss
        :param similarity_matrix: batch_size X (n_contrastive_samples+1)
        :param positive_on_diagonal: if positive samples are on the diagonal, otherwise its in column 0
        :return: loss, top k accuracy
        """

        # the eye positions are the correct classifications of z_next and z_next_hat
        loss = -1 * nn.LogSoftmax(dim=1)(similarity_matrix / self.temperature)

        # aggregate loss - take only -log probabilities of positive samples
        loss = torch.diagonal(loss).mean() if positive_on_diagonal else loss[:, 0].mean()

        # compute accuracy
        accuracy = {}
        for k in self.top_k:
            _, preds = similarity_matrix.topk(k=k, dim=1)
            labels = torch.arange(similarity_matrix.size()[0]).repeat((k, 1)).t().to(self.device) if positive_on_diagonal else torch.zeros_like(preds)
            accuracy[k] = preds.eq(labels).float().mean() * k

        return loss, accuracy

    @staticmethod
    def dot_product_similarity(z1: Tensor, z2: Tensor) -> Tensor:
        """ dot product similarity """
        return torch.mm(z1, z2.t())

    @staticmethod
    def cosine_similarity(z1: Tensor, z2: Tensor) -> Tensor:
        """ cosine similarity """
        z1 = z1.div(z1.norm(p=2, dim=-1, keepdim=True))
        z2 = z2.div(z2.norm(p=2, dim=-1, keepdim=True))
        return torch.mm(z1, z2.t())

    @staticmethod
    def mse_similarity(z1: Tensor, z2: Tensor) -> Tensor:
        """ MSE similarity """
        return -1. * ((z1 ** 2).sum(-1).unsqueeze(1) - 2 * torch.mm(z1, z2.t()) + (z2 ** 2).sum(-1).unsqueeze(0))

    def forward(self, observation: Tensor, next_observation: Tensor, action: Tensor):
        """
        :param observation: input observation image at time t
        :param next_observation: input observation image at time t+1
        :param action: action taken at time t
        :return: z - representation at time t, z_next - representation at time t+1,
                 z_next_hat - predicted representation at time t+1 from representation at time t
        """
        z = self.encoder(observation)
        z_next = self.encoder(next_observation)
        z_next_hat = self.forward_model(z, action)
        return z, z_next, z_next_hat