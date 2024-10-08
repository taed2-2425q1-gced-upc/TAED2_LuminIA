import cv2
import os

image_paths = [
    '/home/alumne/TAED2_LuminIA/data/raw/ts/00626.jpg',
    '/home/alumne/TAED2_LuminIA/data/raw/ts/00133.jpg',
    '/home/alumne/TAED2_LuminIA/data/raw/ts/00123.jpg',
]

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
    else:
        print(f"Loaded image successfully: {img_path}")
