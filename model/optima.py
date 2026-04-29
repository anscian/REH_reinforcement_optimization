import torch
from model import PropertyPredictor
import numpy as np
from PIL import Image, ImageFilter
import json

import importlib

run = 'no_scaler'
checkpoint = torch.load(f'runs/{run}/best.pth')
model = PropertyPredictor(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

device = next(model.parameters()).device
print(model.eval())

for p in model.parameters():
    p.requires_grad = False

has_scaler = checkpoint['config']['mlp_head']['scaler_dim'] > 0

def penalty(image : torch.tensor, x : torch.tensor, **params) -> float:
    '''
    A penalty score for the given input based on its strength and auxeticity retention
    ---------------------------------------------------------------------------------------
    Formula: scale * (-alpha * (E/Es) + beta ** (v + 1) + gamma * v)
    Parameters:
        image : The image torch.Tensor
        x     : Tensor that represents the output E/Es, v of the input reinforcement (1, 2)
    Return:
        The penalty score
    '''
    return params['scale'] * (
        -params['alpha'] * x[:, 0]
        +params['beta'] ** (x[:, 1] + 1)
        +params['gamma'] * x[:, 1]
    )

image, scaler, _ = model.train_loader.dataset[torch.randint(0, len(model.train_loader.dataset), (1,)).item()]
# image, scaler = torch.randn_like(image) * 255.0, torch.randn_like(scaler)
image = torch.ones_like(image) * (0.5)
# image = torch.tensor(np.array(model.inverse_transform(Image.open('start.png')))).unsqueeze(0)
# image = image / 255.0
start = image.clone()
image = image.unsqueeze(0).to(device).requires_grad_(True)
scaler = scaler.unsqueeze(0).to(device).requires_grad_(has_scaler)
print('Starting with:', 'Image', image, 'Scaler', scaler)

optimizer = torch.optim.Adam([image, scaler] if has_scaler else [image], lr=0.01)

steps = 1000
params = {
    'alpha' : 30.0,
    'beta'  : 25.0,
    'gamma' : 4.0,
    'scale' : 100,
}

best_loss = float('inf')
best_image = None
for step in range(steps):
    im = 255.0 * torch.sigmoid(image)

    pred = model(im, scaler)
    loss = penalty(im, pred, **params)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([image], 1.0)
    optimizer.step()

    print(
        f'Step {step:03d} | '
        f'Loss: {loss.item():.4f} | '
        f'E/Es: {pred[:,0].item():.4f} | '
        f'grad: {image.grad.abs().mean():.4f} | '
        f'v: {pred[:,1].item():.4f}'
    )

    image.data += 0.005 * torch.randn_like(image)

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_image = im.detach().clone()

image = best_image.detach().cpu().squeeze().numpy()
Image.fromarray(image.astype(np.uint8), mode='L').save('tmp.png')
image = (image > 127).astype(np.uint8) * 255
image = Image.fromarray(image, mode='L')
image = model.inverse_transform(image)
# from src.image_transformations import bilinear_upsample
# image = bilinear_upsample(image, model.config['dataset']['raw_image_size'])
# image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
image.save(f'runs/{run}/optimized_reinforcement_image.png')

start = (255.0 * torch.sigmoid(start)).detach().cpu().squeeze().numpy()
start = (start > 127).astype(np.uint8) * 255
start = Image.fromarray(start, mode='L')
start = model.inverse_transform(start)
start.save('start.png')

if checkpoint['config']['mlp_head']['scaler_dim']:
    scaler = scaler.detach().cpu().squeeze().numpy().tolist()
    scaler = {'n' : scaler[0], 'r' : scaler[1], 'VF' : scaler[2]}
    with open(f'runs/{run}/optimized_reinforcement_config.json', 'w') as f:
        json.dump(scaler, f, indent=4)