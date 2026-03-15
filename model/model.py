from src.dataloader import load_data
from src.mlp_head import MLPHead

from torch.utils.data import DataLoader
import torch
import torch.nn as nn


######################### CONIFGURABLE ################################
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

IMAGES_DIR  = './dataset/images'
DATA_PATH   = './dataset/results.csv'
IMG_EXT     = '.png'
BATCH_SIZE  = 16
NUM_WORKERS = 0

from src.image_encoder.resnet import ResnetEncoder as ImageEncoder
IMAGE_ENCODER_INIT_PARAMS = [18], {'pre_trained' : True, 'freeze' : True}

SCALER_DIM = 3
MLP_ARCHITECTURE = [
        {'out_dim' : 512, 'activation' : 'GELU', 'dropout' : 0.15},
        {'out_dim' : 256, 'activation' : 'GELU', 'dropout' : 0.10},
        {'out_dim' : 128, 'activation' : 'GELU'},
]
######################### CONFIGURABLE #################################


class PropertyPredictor(nn.Module):
    def __init__(self):
        '''
        Initializes the main end-to-end model
        -------------------------------------------------------------------------
        Parameters:
        Returns:
        
        **About output_to_pred**
        - pred[0] is E/Es and thus is sigmoid activation of the output[0]
        - pred[1] is v which is as is the output[1] (output layer is purely linear)
        '''

        super().__init__()
        self.device = DEVICE
        self.device_is_cuda = self.device.type == 'cuda'

        self.train_loader, self.test_loader = map(
            lambda x : DataLoader(
                x,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=self.device_is_cuda
            ),
            load_data(DATA_PATH, IMAGES_DIR, IMG_EXT, train_test_split=True, transform=None)
        )

        self.image_encoder = ImageEncoder(*IMAGE_ENCODER_INIT_PARAMS[0], **IMAGE_ENCODER_INIT_PARAMS[1])

        self.mlp_head = MLPHead(self.image_encoder.get_encoding_dim(), SCALER_DIM, MLP_ARCHITECTURE)

        self.output_to_pred = lambda y : torch.cat([
            torch.sigmoid(y[:, :1]),
            y[:, 1:]
        ], dim=1)

    
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


    def train_model(self, lr : float = 1e-3, loss_fn : callable = nn.SmoothL1Loss(), epochs : int = 10, verbose : bool = True):
        '''
        Training loop for the end-to-end REH Stiffness and Poisson's Ratio predictor
        ----------------------------------------------------------------------------
        Parameters:
            lr      : Learning rate [default .001]
            loss_fn : Loss function [default torch.nn.SmoothL1Loss()]
            epochs  : Number of training epochs [default 100]
            verbose : Display info after every epoch
        Returns:
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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


    def test_model(self, loss_fn : callable = nn.SmoothL1Loss()) -> float:
        '''
        Run predictions on test data and compute the loss over ground truth
        ---------------------------------------------------------------------
        Parameters:
            loss_fn : The loss function to be used to compute loss on test set
        Returns:
            Loss computed over test set over ground truth
        '''

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
    model = PropertyPredictor().to(DEVICE)
    print(model.eval(), end='\n\n')
    model.train_model()
    print('Test loss : ', model.test_model())
