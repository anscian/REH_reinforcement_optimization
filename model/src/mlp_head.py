from collections import OrderedDict
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, image_encoding_dim : int, scaler_dim : int, archi : list[dict]) -> None:
        '''
        Initializes the MLP head that utilizes image embeddings and other metadata for output vector
        ----------------------------------------------------------------------------------------------------------------
        Parameters:
            image_encoding_dim : E, the size of image encoding
            scaler_dim         : S, the size of metadata vector (handles 0 too)
            archi              : A list of dictionaries each describing ONLY the hidden layers of FCNN
                                 
                                 **
                                 Output layer is known to be linear on last hidden layer with no activation and
                                 dropout with 2 dimensional output.
                                 Initial input dimension is set to be I (= E + S)
                                 **
                                 
                                 Format: {
                                    `out_dim`    : ...,
                                    `activation` : `GELU`, `ReLU`, etc. (raises error if invalid, i.e., not in torch.nn)
                                                   - None or not specified means not used
                                    `dropout`    : p between 0 and 1 (raises error if not) 
                                                   - None or not specified means not used
                                 }
        Returns:
        '''

        super().__init__()

        self.has_scaler = scaler_dim > 0

        archi = archi.copy()
        archi.append({'out_dim' : 2})
        archi[0]['in_dim'] = image_encoding_dim + scaler_dim
        for i in range(1, len(archi)):
            archi[i]['in_dim'] = archi[i - 1]['out_dim']

        layers = OrderedDict()

        for i, layer in enumerate(archi):
            block = []

            block.append(nn.Linear(layer['in_dim'], layer['out_dim']))

            if (act := layer.get('activation')) is not None:
                block.append(getattr(nn, act)())

            if (p := layer.get('dropout')) is not None:
                block.append(nn.Dropout(p))
            
            layers[f'layer{i}'] = nn.Sequential(*block)
        
        self.fcnn = nn.Sequential(layers)


    def forward(self, img_enc : torch.Tensor, scaler : torch.Tensor) -> torch.Tensor:
        '''
        Forward Propagation to get prediction from image encoding and scalers
        ---------------------------------------------------------------------
        Parameters:
            img_enc : torch.Tensor (B, E)
            scaler  : torch.Tensor (B, S)
        Returns:
            Torch tensor (B, 2) from which predictions would be derived
        '''

        x = torch.cat([img_enc, scaler], dim=1) if self.has_scaler else img_enc
        return self.fcnn(x)
