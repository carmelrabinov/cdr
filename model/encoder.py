from typing import List

import torch
from torch import nn, Tensor
import torchvision.models as models


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._image_size = input_dim[0]

        # generate network
        self.features_model = self.generate_net()

        # calculate output features size and generate last fc layer
        self.generate_output_layer(self.input_dim, self.output_dim)

    def generate_net(self) -> nn.Module:
        """ generates encoder model """
        raise NotImplementedError

    @property
    def image_size(self) -> int:
        """ input image size """
        return self._image_size

    @image_size.setter
    def image_size(self, image_size):
        """ input image size """
        self._image_size = image_size

    @property
    def name(self):
        """ encoder name """
        return NotImplementedError

    def extract_features(self, x, flatten: bool = True) -> Tensor:
        """ encode input image to features """
        x = self.features_model(x)
        if flatten:
            x = Flatten()(x)
        return x

    def forward(self, x):
        x = self.features_model(x)
        x = Flatten()(x)
        x = self.out_fc(x)
        return x

    def generate_output_layer(self, input_dim: List[int], output_dim: int) -> None:
        """
        calculate the features size W.R.T the input image and generate FC layer the project features to the wanted
        latent size
        :param input_dim: input image size
        :param output_dim: wanted latent size
        :return: None
        """
        # calculate output features size
        x = self.features_model(torch.randn((1, 3, input_dim[0], input_dim[1])))
        features_size = x.view(-1).size()[0]

        self.out_fc = nn.Linear(features_size, output_dim, bias=False)
        self.n_params = sum(p.numel() for p in self.parameters())

        print(f"Running with {self.name} Encoder with {self.n_params} parameters, {features_size} features, projected to {output_dim}")


class Resnet18Encoder(Encoder):
    @property
    def name(self):
        return "Resnet-18"

    def generate_net(self):

        # load initialized model
        pre_trained_encoder = models.resnet18(pretrained=True)

        # remove last fc layer but include the max pool layer
        return nn.Sequential(*list(pre_trained_encoder.children())[:-1])


class ConvNetFMEncoder(Encoder):
    @property
    def name(self):
        return "ConvNet"

    def generate_net(self):
        model = nn.Sequential(
                nn.Conv2d(self.input_dim[2], 64, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 64 x 32 x 32
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 16 x 16
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # Option 1: 256 x 8 x 8
                nn.Conv2d(256, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 256 x 4 x 4
                )
        return model


class LiteConvNetEncoder(Encoder):
    @property
    def name(self):
        return "LiteConvNet"

    def generate_net(self):
        model = nn.Sequential(
                nn.Conv2d(self.input_dim[2], 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 64 x 64
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 32 x 32
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 16 x 16
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 8 x 8
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 4 x 4
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 128 x 2 x 2
                )
        return model



