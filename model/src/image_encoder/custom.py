import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class ConvEncoder(nn.Module):
    def __init__(self, archi : list[dict]) -> None:
        '''
        Initializes a customized image encoder whose architecture is provided in order-wise feed forward fashion
        ----------------------------------------------------------------------------------------------------------------
        Parameters:
            archi : A list of dictionaries each describing ONLY the hidden layers of FCNN
                                 
                    **
                    Input is known to have 1 channel only
                    **
                    
                    Format: {
                        `out_channels` : ...,
                        `kernel_size`  : ...,
                        `stride`       : ...,
                        `padding`      : ...,
                        `activation`   : `GELU`, `ReLU`, etc. (raises error if invalid, i.e., not in torch.nn)
                                         - None or not specified means not used
                        `batch_norm`   : True or False
                    }
        Returns:
        '''

        super().__init__()

        archi[0]['in_channels'] = 1
        for i in range(1, len(archi)):
            archi[i]['in_channels'] = archi[i - 1]['out_channels']

        layers = OrderedDict()

        for i, layer in enumerate(archi):
            block = []

            block.append(nn.Conv2d(
                layer['in_channels'], layer['out_channels'], 
                kernel_size=layer['kernel_size'],
                stride=layer['stride'],
                padding=layer['padding'] 
            ))

            if layer['batch_norm']:
                block.append(nn.BatchNorm2d(layer['out_channels']))

            if (act := layer.get('activation')) is not None:
                block.append(getattr(nn, act)())
            
            layers[f'layer{i}'] = nn.Sequential(*block)

        layers['pool'] = nn.AdaptiveAvgPool2d((1, 1))
        
        self.encoder = nn.Sequential(layers)

        self.encoding_size = archi[-1]['out_channels']


    def forward(self, imgs : torch.Tensor) -> torch.tensor:
        '''
        Forward Propagation
        ----------------------------------------------------------------------
        Paramters:
            imgs : torch.Tensor (B, 1, H, L)
        Returns:
            Torch tensor result after forward propagation (encoding) -> (B, E)
        '''

        return torch.flatten(self.encoder(imgs / 255.0), 1)


    def get_encoding_dim(self) -> int:
        '''
        Provides final encoding size of the Image Encoder
        -------------------------------------------------
        Parameters:
        Returns:
            The output image encoding dimension E
        '''

        return self.encoding_size