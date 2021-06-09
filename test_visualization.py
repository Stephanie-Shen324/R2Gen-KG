#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from skimage.transform import resize
from PIL import Image
import imageio
import json
import argparse
import gzip
import math
import os
import pickle
import random
import sys
import time
import traceback
import nltk
import numpy as np
import torch.utils.data
import torch.distributed as dist
from torchvision.datasets.folder import default_loader


import sys

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def visualize_att(img_path, seq, alphas,  smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(img_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    # image=image[0]
    # image = np.transpose(image, (1, 2, 0))
    # image = Image.fromarray(image, 'RGB')
    words = seq.split()

    for t in range(min(len(words),alphas.shape[0])):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)

        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig('test.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--images_id', default='results/iu_xray/test_e1_images_id.json', help='path to image id')
    parser.add_argument('--att_alpha',  default='results/iu_xray/test_e1_att_alpha.json', help='path to attention weight')
    parser.add_argument('--res', default='results/iu_xray/test_e1_res.json', help='path to generete report')
    parser.add_argument('--images_path',  default='data/iu_xray/images', help='path to image')
    parser.add_argument('--num', type=int,  default=0, help='the number n of pic want to show')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()


    f = open(args.images_id)
    data = json.load(f)
    img_id = data[args.num]
    img_path=os.path.join('data/iu_xray/images',img_id,'0.png')

    f = open(args.att_alpha)
    data = json.load(f)
    att_alpha = np.array(data)[args.num]

    f = open(args.res)
    data = json.load(f)
    sentence = data[args.num]

    # Visualize caption and attention of best sequence
    visualize_att(img_path, sentence, att_alpha,  args.smooth)
