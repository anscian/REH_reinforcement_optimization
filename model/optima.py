import torch
from model import PropertyPredictor
import numpy as np
from PIL import Image, ImageFilter
import json

import importlib

checkpoint = torch.load('best_no_scaler.pth')
model = PropertyPredictor(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

device = next(model.parameters()).device
print(model.eval())

for p in model.parameters():
    p.requires_grad = False

def penalty(image : torch.tensor, x : torch.tensor, **params) -> float:
    '''
    A penalty score for the given input based on its strength and auxeticity retention
    ----------------------------------------------------------------------------------
    Formula: - alpha * (E/Es) - gamma * log(relu(-v) + eps) + beta * relu(v)^2
    Parameters:
        image : The image torch.Tensor
        x     : Tensor that represents the output E/Es, v of the input reinforcement (1, 2)
    Return:
        The penalty score
    '''
    return (
        -params['alpha'] * x[:, 0]
        -params['gamma'] * torch.log(torch.relu(-x[:, 1]) + params['eps'])
        +params['beta'] * torch.relu(x[:, 1])**2
    )

image, scaler, _ = model.train_loader.dataset[torch.randint(0, len(model.train_loader.dataset), (1,)).item()]
image, scaler = (torch.rand_like(image) > 0.5) * 255.0, torch.rand_like(scaler)
image = image.unsqueeze(0).to(device).requires_grad_(True)
scaler = scaler.unsqueeze(0).to(device).requires_grad_(True)
print('Starting with:', 'Image', image, 'Scaler', scaler)

optimizer = torch.optim.Adam([image, scaler], lr=0.05)

steps = 50
params = {
    'alpha' : 30000.0,
    'gamma' : 2.0,
    'beta'  : 5.0,
    'eps'   : 1e-4,
}

for step in range(steps):
    pred = model(image, scaler)
    loss = penalty(image, pred, **params)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    image.data.clamp_(0, 1)

    if (step + 1) % 1 == 0:
        print(
            f'Step {step:03d} | '
            f'Loss: {loss.item():.4f} | '
            f'E/Es: {pred[:,0].item():.4f} | '
            f'v: {pred[:,1].item():.4f} | '
        )

image = image.detach().cpu().squeeze().numpy()
image = (image > 0.5).astype(np.uint8) * 255
image = Image.fromarray(image, mode='L')
image = model.inverse_transform(image)
# from src.image_transformations import bilinear_upsample
# image = bilinear_upsample(image, model.config['dataset']['raw_image_size'])
# image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
image.save('optimized_reinforcement_image_no_scaler.png')

if checkpoint['config']['mlp_head']['scaler_dim']:
    scaler = scaler.detach().cpu().squeeze().numpy().tolist()
    scaler = {'n' : scaler[0], 'r' : scaler[1], 'VF' : scaler[2]}
    with open('optimized_reinforcement_config.json', 'w') as f:
        json.dump(scaler, f, indent=4)