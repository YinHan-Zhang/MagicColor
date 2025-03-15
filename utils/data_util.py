import numpy as np
import cv2
from PIL import Image, ImageDraw
import random

# Function to generate a random mask
def random_mask(im_shape, mask_type='random', **kwargs):
    if mask_type == 'random':
        return generate_random_mask(im_shape, **kwargs)
    elif mask_type == 'irregular':
        return make_irregular_mask(*im_shape, **kwargs)
    elif mask_type == 'rectangle':
        return make_rectangle_mask(*im_shape, **kwargs)
    elif mask_type == 'uncrop':
        return make_uncrop(*im_shape, **kwargs)
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'")

# Function to generate a random rectangular or elliptical mask
def generate_random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return np.array(mask)

# Function to generate irregular mask
def make_irregular_mask(w, h, max_angle=4, max_length=200, max_width=100, min_strokes=1, max_strokes=5, mode='line'):
    # initialize mask
    assert mode in ['line', 'circle', 'square']
    mask = np.zeros((h, w), np.float32)

    # draw strokes
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        x1 = np.random.randint(w)
        y1 = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 60 + np.random.randint(max_length)
            radius = 20 + np.random.randint(max_width)
            x2 = np.clip((x1 + length * np.sin(angle)).astype(np.int32), 0, w)
            y2 = np.clip((y1 + length * np.cos(angle)).astype(np.int32), 0, h)
            if mode == 'line':
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, radius)
            elif mode == 'circle':
                cv2.circle(mask, (x1, y1), radius=radius, color=1.0, thickness=-1)
            elif mode == 'square':
                radius = radius // 2
                mask[y1 - radius:y1 + radius, x1 - radius:x1 + radius] = 1
            x1, y1 = x2, y2
    return mask

# Function to generate rectangle mask
def make_rectangle_mask(w, h, margin=10, min_size=30, max_size=150, min_strokes=1, max_strokes=4):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # draw rectangles
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        box_w = np.random.randint(min_size, max_size)
        box_h = np.random.randint(min_size, max_size)
        x1 = np.random.randint(margin, w - margin - box_w + 1)
        y1 = np.random.randint(margin, h - margin - box_h + 1)
        mask[y1:y1 + box_h, x1:x1 + box_w] = 1
    return mask

# Function to generate uncrop mask
def make_uncrop(w, h):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # randomly halve the image
    side = np.random.choice([0, 1, 2, 3])
    if side == 0:
        mask[:h // 2, :] = 1
    elif side == 1:
        mask[h // 2:, :] = 1
    elif side == 2:
        mask[:, :w // 2] = 1
    elif side == 3:
        mask[:, w // 2:] = 1
    return mask

def laplacian_variance(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute the variance of the Laplacian
    variance = laplacian.var()
    
    return variance