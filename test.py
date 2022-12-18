import argparse
import glob

import numpy as np

from PIL import Image

import time
import os

from networks_trained import *

# testing all images by encoding and decoding, calculate MSE and bpp
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--qp", default=1, type=int,
        help="Quality parameter, choose from [1~7] (model0) or [1~8] (model1)"
    )
    parser.add_argument(
        "--model_type", default=0, type=int,
        help="Model type, choose from 0:PSNR 1:MS-SSIM"
    )
    parser.add_argument(
        "--save_recon", default=0, type=int,
        help="Whether to save reconstructed image in the encoding process."
    )
    # argument: --with_iar
    parser.add_argument(
        "--with_iar", default=1, type=int,
        help="Whether to use IAR in the encoding process. 0: no IAR, 1: IAR"
    )
    parser.add_argument(
        "--device", default='cpu', type=str,
        help="Which device does the network run on?"
    )
    parser.add_argument('--images', type=str)
    args = parser.parse_args()

    # Load images
    images = glob.glob(os.path.join(args.images, '*.png'))

    errors = []
    bpp = []
    for image in images:
        args.input = image
        args.output = image.replace('.png', '.bin')
        print("Compressing: " + image)
        compress_low(args)

        args.input = image.replace('.png', '.bin')
        args.output = image.replace('.png', 'recover.png')
        print("Decompressing: " + image)
        decompress_low(args)

        img = Image.open(image)
        img_recover = Image.open(image.replace('.png', 'recover.png'))
        # error = mse
        error = np.mean((np.array(img) - np.array(img_recover)) ** 2)
        errors.append(error)
        bin_size = os.path.getsize(image.replace('.png', '.bin'))
        bpp.append(bin_size * 8 / img.width / img.height)
    
    print("MSE: ", np.mean(errors))
    print("BPP: ", np.mean(bpp))