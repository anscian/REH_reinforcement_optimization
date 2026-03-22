import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class ResnetEncoder(nn.Module):
    def __init__(self, param_layers : int = 18, pre_trained : bool = False, freeze : bool = True) -> None:
        '''
        Initialize Standard Resnet Architecture (18 or 34 layers)
        ------------------------------------------------------------
        Parameters:
            param_layers : number of parameter layes [18 or 34]
            pre_trained  : whether pre-trained weights are used
            freeze       : the model weights are frozen for training
        Returns:
        '''

        super().__init__()
        assert param_layers in (18, 34), 'Only 18 and 34 layer standard Resnets available'
        model   = resnet18 if param_layers == 18 else resnet34
        weights = ResNet18_Weights.DEFAULT if param_layers == 18 else ResNet34_Weights.DEFAULT
        
        self.encoder = model(weights=weights if pre_trained else None)

        _conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=_conv1.out_channels,
                kernel_size=_conv1.kernel_size,
                stride=_conv1.stride,
                padding=_conv1.padding,
                bias=_conv1.bias
        )
        with torch.no_grad():
            self.encoder.conv1.weight[:] = _conv1.weight.mean(dim=1, keepdim=True)

        self.encoding_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        if freeze:
            for param in self.encoder.conv1.parameters():
                param.requires_grad = False
            for param in self.encoder.layer1.parameters():
                param.requires_grad = False
        self.frozen_encoder = freeze


    def forward(self, imgs : torch.Tensor) -> torch.tensor:
        '''
        Forward Propagation
        ----------------------------------------------------------------------
        Paramters:
            imgs : torch.Tensor (B, 1, H, L)
        Returns:
            Torch tensor result after forward propagation (encoding) -> (B, E)
        '''

        return self.encoder(imgs / 255.0)


    def get_encoding_dim(self) -> int:
        '''
        Provides final encoding size of the Image Encoder
        -------------------------------------------------
        Parameters:
        Returns:
            The output image encoding dimension E
        '''

        return self.encoding_size
