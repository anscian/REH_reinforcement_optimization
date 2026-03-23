from src.dataloader import load_data
from src.mlp_head import MLPHead

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import importlib


class PropertyPredictor(nn.Module):
    def __init__(self, config : dict) -> None:
        '''
        Initializes the main end-to-end model
        ---------------------------------------------------------------------------
        Parameters:
            config : Contains final model's configuration (structure given above)
        Returns:
        
        **About output_to_pred**
        - pred[0] is E/Es and thus is sigmoid activation of the output[0]
        - pred[1] is v which is as is the output[1] (output layer is purely linear)
        '''

        super().__init__()
        self.config = config
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.device_is_cuda = self.device.type == 'cuda'

        self.transform_fn = self.__load_object(config['image_sampling']['transform'])
        self.transform = lambda im : self.transform_fn(im, self.config['dataset']['transformed_size'])
        self.inverse_transform_fn = self.__load_object(config['image_sampling']['inverse_transform'])
        self.inverse_transform = lambda im : self.inverse_transform_fn(im, self.config['dataset']['raw_image_size'])

        self.train_loader, self.test_loader = map(
            lambda x : DataLoader(
                x,
                batch_size=config['dataset']['batch_size'],
                shuffle=True,
                num_workers=config['dataset']['num_workers'],
                pin_memory=self.device_is_cuda
            ),
            load_data(
                config['dataset']['data_path'], 
                config['dataset']['images_dir'], 
                config['dataset']['img_ext'], 
                train_test_split=True, transform=self.transform
            )
        )

        self.image_encoder = self.__load_object(config['image_encoder']['name'])(
            *config['image_encoder']['args'], 
            **config['image_encoder']['kwargs']
        )

        self.mlp_head = MLPHead(
            self.image_encoder.get_encoding_dim(), 
            config['mlp_head']['scaler_dim'], 
            config['mlp_head']['architecture']
        )

        self.output_to_pred = lambda y : torch.cat([
            torch.sigmoid(y[:, :1]),
            y[:, 1:]
        ], dim=1)

        self.to(self.device)


    @staticmethod
    def __load_object(name : str) -> object:
        module_path, class_name = name.rsplit('.', 1)
        return getattr(importlib.import_module(module_path), class_name)

    
    def forward(self, img_tensor : torch.Tensor, scaler_tensor : torch.Tensor) -> torch.Tensor:
        '''
        Forward Propagation
        ---------------------------------------------------------------
        Takes img_tensor and scaler tensor, and gives prediction (B, 2)
        Parameters:
            img_tensor    : torch.Tensor (B, 1, H, L) batch of images
            scaler_tensor : torch.Tensor (B, 3) batch of scalers
        Returns:
            torch.Tensor [E/Es, v] (B, 2)
        '''

        img_enc = self.image_encoder(img_tensor)
        y = self.mlp_head(img_enc, scaler_tensor)
        return self.output_to_pred(y)


    def train_model(self, verbose : bool = True) -> None:
        '''
        Training loop for the end-to-end REH Stiffness and Poisson's Ratio predictor
        ----------------------------------------------------------------------------
        Parameters:
            verbose : Display info after every epoch
        Returns:
        '''

        optimizer = self.__load_object(self.config['train']['optimizer']['name'])([
            {'params' : self.image_encoder.parameters(), **self.config['train']['optimizer']['image_encoder']},
            {'params' : self.mlp_head.parameters(), **self.config['train']['optimizer']['mlp_head']}
        ])
        epochs = self.config['train']['epochs']
        loss_fn = self.__load_object(self.config['train']['loss_fn'])()

        self.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for img_tensor, scaler_tensor, target_tensor in self.train_loader:
                img_tensor = img_tensor.to(self.device, non_blocking=self.device_is_cuda)
                scaler_tensor = scaler_tensor.to(self.device, non_blocking=self.device_is_cuda)
                target_tensor = target_tensor.to(self.device, non_blocking=self.device_is_cuda)

                optimizer.zero_grad()
                loss = loss_fn(self(img_tensor, scaler_tensor), target_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}')

        self.train(False)


    def test_model(self) -> float:
        '''
        Run predictions on test data and compute the loss over ground truth
        ---------------------------------------------------------------------
        Parameters:
        Returns:
            Loss computed over test set over ground truth
        '''

        loss_fn = self.__load_object(self.config['train']['loss_fn'])()

        with torch.no_grad():
            total_loss = 0.0

            for img_tensor, scaler_tensor, target_tensor in self.test_loader:
                img_tensor = img_tensor.to(self.device, non_blocking=self.device_is_cuda)
                scaler_tensor = scaler_tensor.to(self.device, non_blocking=self.device_is_cuda)
                target_tensor = target_tensor.to(self.device, non_blocking=self.device_is_cuda)

                loss = loss_fn(self(img_tensor, scaler_tensor), target_tensor)
                total_loss += loss.item()

        return total_loss


if __name__ == '__main__':
    CONFIG = {
        'device' : 'cpu',
        'dataset' : {
            'images_dir'       : './dataset/images',
            'data_path'        : './dataset/results.csv',
            'img_ext'          : '.png',
            'batch_size'       : 16,
            'num_workers'      : 0,
            'raw_image_size'   : (2381, 1576),
            'transformed_size' : (224, 148),
        },
        'image_sampling' : {
            'transform'         : 'src.image_transformations.box_downsample',
            'inverse_transform' : 'src.image_transformations.nearest_upsample',
        },
        'image_encoder' : {
            'name' : 'src.image_encoder.custom.ConvEncoder',
            'args' : [[
                {'out_channels' : 16, 'kernel_size' : 3, 'stride' : 2, 'padding' : 1, 'batch_norm' : True, 'activation' : 'ReLU'},
                {'out_channels' : 32, 'kernel_size' : 3, 'stride' : 2, 'padding' : 1, 'batch_norm' : True, 'activation' : 'ReLU'},
            ]],
            'kwargs' : {},
        },
        'mlp_head' : {
            'architecture' : [
                {'out_dim' : 16, 'activation' : 'GELU', 'dropout' : 0.10},
                {'out_dim' : 8, 'activation' : 'GELU'},
            ],
            'scaler_dim' : 3,
        },
        'train' : {
            'optimizer' : {
                'name' : 'torch.optim.Adam',
                'image_encoder' : {'lr' : 1e-3},
                'mlp_head'      : {'lr' : 1e-3},
            },
            'epochs'  : 50,
            'loss_fn' : 'torch.nn.SmoothL1Loss'
        }
    }

    model = PropertyPredictor(config=CONFIG)
    print(model.eval(), end='\n\n')

    print('Number of trainable parameters:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print('Train data size:', len(model.train_loader.dataset))
    print('Test data size:', len(model.test_loader.dataset), end='\n\n')

    print('Training started')
    model.train_model()
    print('Training complete', end='\n\n')

    model_save_path = 'best.pth'
    print(f'Saving model to {model_save_path}')
    torch.save({
        'model_state_dict' : model.state_dict(),
        'config' : model.config,
    }, model_save_path)
    print('Model saved', end='\n\n')

    print('Testing model on test set')
    print('Test loss : ', model.test_model())