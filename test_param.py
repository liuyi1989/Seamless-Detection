import sys
import importlib
from data import Test_Dataset
#from data_esod import ESOD_Test

import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
import importlib

from base.framework_factory import load_framework
from metric import *
from thop import profile
#from framework_factory import load_framework

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, schedule = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    model = importlib.import_module('base.model').Network(net_name, config)
    model = model.cuda()

    
    input = torch.randn(1, 3, config['size'], config['size']).cuda()
    flops, _ = profile(model, inputs=(input, ))
    params = params_count(model)
    
    print('FLOPs:')
    print(flops)
    print('Params:')
    print(params)
 
    print('FLOPs(1e9): {:.2f}, Params(1e6): {:.2f}.'.format(flops / 1e9, params / 1e6))
    
        
if __name__ == "__main__":
    main()