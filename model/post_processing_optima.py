from PIL import Image
import cv2
import numpy as np

run = 'very_small'
im = Image.open(f'runs/{run}/optimized_reinforcement_image.png')
im = np.array(im)

kernel = np.ones((8, 8), np.uint8)
im = cv2.GaussianBlur(im, (0, 0), sigmaX=5)
im = (im > 0.2).astype(np.uint8) * 255
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

Image.fromarray(im).save(f'runs/{run}/processed_reinforcement.png')