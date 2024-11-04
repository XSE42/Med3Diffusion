from typing import Tuple

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class NumCondEncoder(ModelMixin, ConfigMixin):
    r"""
    A numeric condition encoder that encodes the numeric condition into a condition embedding.

    This is a MLP model, the shapes are:
        input `(batch_size, in_features)`
        hidden `(batch_size, out_channels, hidden_features)`
        output `(batch_size, out_channels, out_features)`

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models.

    Parameters:
        in_features (int, *optional*, defaults to 2): Number of features in the input.
        out_channels (int, *optional*, defaults to 32): Number of channels in the output.
        out_features (int, *optional*, defaults to 64): Number of features in the output.
        hidden_features (int, *optional*, defaults to 16): Number of hidden features in the hidden layer.
        num_hidden_layers (int, *optional*, defaults to 1): Number of hidden layers in MLP.
        norm_num_groups (int, *optional*, defaults to 8): Number of groups in the normalization layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_features: int = 2,
        out_channels: int = 32,
        out_features: int = 64,
        hidden_features: int = 16,
        num_hidden_layers: int = 1,
        norm_num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.linear_in = nn.Linear(in_features, out_channels * hidden_features)

        self.layers = nn.ModuleList([])
        self.layers.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels))
        self.layers.append(nn.SiLU())
        self.layers.append(nn.Dropout(p=dropout))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encodes the numeric condition into a condition embedding.

        Args:
            x (torch.Tensor): The numeric condition tensor of shape `(batch_size, in_features)`.

        Returns:
            torch.Tensor: The condition embedding of shape `(batch_size, out_channels, out_features)`.
        """
        x = self.linear_in(x)
        x = x.view(x.shape[0], self.out_channels, self.hidden_features)

        for layer in self.layers:
            x = layer(x)

        return x


class AtlasLatentModel(ModelMixin, ConfigMixin):
    r"""
    A model that generates the atlas noise latent

    Parameters:
        latent_sample_size: (H, W) or (D, H, W), the sample size of atlas noise latent
        latent_channel: C, the number of channels of atlas noise latent

    Outputs:
        atlas noise latent with shape (1, latent_channel, *latent_sample_size)
    """

    @register_to_config
    def __init__(self, latent_sample_size, latent_channel):
        super().__init__()

        latent_shape = (1, latent_channel, *latent_sample_size)
        latent = torch.randn(latent_shape)
        self.latent = nn.Parameter(latent, requires_grad=True)

    def forward(self):
        r"""
        Generate the atlas noise latent
        """
        return self.latent


class VxmCondAtlas(ModelMixin, ConfigMixin):
    r"""
    Conditional atlas generator in Voxelmorph, which takes the numeric condition as input and generates the atlas.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models.

    Parameters:
        cond_features (int, *optional*, defaults to 1): Number of features in the numeric condition.
            In Voxelmorph, this is named as `pheno_input_shape` in `ConditionalTemplateCreation`.
        atlas_shape (Tuple[int, int, int], *optional*, defaults to (112, 128, 112)): Shape of the atlas.
            In Voxelmorph, this is named as `inshape` in `ConditionalTemplateCreation`.
        out_channels (int, *optional*, defaults to 1): Number of channels in the atlas.
            In Voxelmorph, this is named as `atlas_feats` in `ConditionalTemplateCreation`.
        hidden_channels (int, *optional*, defaults to 4): Number of channels in the hidden layers.
            In Voxelmorph, this is named as `conv_nb_features` in `ConditionalTemplateCreation`.
        extra_conv_layers (int, *optional*, defaults to 3): Number of extra convolutional layers in the model.
            In Voxelmorph, this is named as `extra_conv_layers` in `ConditionalTemplateCreation`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        cond_features: int = 1,
        atlas_shape: Tuple[int, int, int] = (112, 128, 112),
        out_channels: int = 1,
        hidden_channels: int = 4,
        extra_conv_layers: int = 3,
    ):
        super().__init__()

        self.atlas_shape = torch.tensor(atlas_shape)
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.in_dense = nn.Linear(cond_features, hidden_channels * torch.prod(self.atlas_shape))
        self.in_conv = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1)

        self.extra_convs = nn.ModuleList([])
        for _ in range(extra_conv_layers):
            self.extra_convs.append(
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1)
            )

        self.out_conv = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=1)
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-7)
        nn.init.normal_(self.out_conv.bias, mean=0.0, std=1e-7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Generates the atlas according to numeric condition.

        Args:
            x (torch.Tensor): The numeric condition tensor of shape `(batch_size, cond_features)`.

        Returns:
            torch.Tensor: The atlas of shape `(batch_size, out_channels, *atlas_shape)`.
        """
        x = self.in_dense(x)
        x = x.view(x.shape[0], self.hidden_channels, *self.atlas_shape)
        x = self.in_conv(x)

        for layer in self.extra_convs:
            x = layer(x)

        x = self.out_conv(x)

        return x

