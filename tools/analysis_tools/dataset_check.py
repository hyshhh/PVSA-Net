import cv2
import numpy as np
import os

label_dir = "./data/cityscapes_cleaned/annotations/val"

for file in os.listdir(label_dir)[:20]:
    path = os.path.join(label_dir, file)

    mask = cv2.imread(path, 0)

    print(file)
    print(np.unique(mask))
    print()