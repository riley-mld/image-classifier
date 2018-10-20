import numpy as np

import argparse
import json
from PIL import Image

import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
import torch.nn.functional as F

import functions as func


def main():
    start_time = time()
    
    in_arg = get_input_args()
    
    device = set_device(in_arg.gpu)
    
    prob, classes = func.predict(in_arg.image_dir, in_arg.checkpoint, device, in_arg.top_k)
    names = []
    try:
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        for clas in classes:
            names.append(cat_to_name[str(clas)])
    except:
        pass
    
    if len(names) != 0:
        for i in range(len(classes)):
            print("Classes: ", classes[i], names[i], " Probabilty: ", prob[i])
    elif not names:
        if len(names) == 0:
            for i in range(len(classes)):
                print("Classes: ", classes[i], " Probabilty: ", prob[i])
    
    
def get_input_args():
    """Get  command line arguments. """
    # Creates parser
    parser = argparse.ArgumentParser()

    parser.add_argument('image_dir', type=str, 
                        help='path to the image')
    parser.add_argument('checkpoint', type=str, 
                        help='path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top K')
    parser.add_argument('--category_names', type=str, default='', 
                        help='file patch to category name file')
    parser.add_argument('--gpu', type=str, default='cpu', 
                        help='set device for training.')
    
    return parser.parse_args()          
          
    
def set_device(gpu):
    """Set device."""
    if gpu.lower() == 'cpu':
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device


main()