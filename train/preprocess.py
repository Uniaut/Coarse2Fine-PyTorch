import argparse
import glob

import numpy as np

from PIL import Image

import time
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--images', type=str)
    args = parser.parse_args()

    # Load images
    images = glob.glob(os.path.join(args.images, '*.png'))

    for image in images:
        print(image)
        img = Image.open(image)
        img_resize = img.resize((int(img.width / 2), int(img.height / 2)))
        img_resize.save(image.replace('.png', '_half.png'), 'png')
