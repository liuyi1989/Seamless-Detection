import sys
import importlib
from data import Test_Dataset
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
from torch import nn
from metric import *
import importlib

from base.framework_factory import load_framework

def test_model(test_sets, config, epoch=None, saver=None):

    st = time.time()
    print("######################")
    for set_name, test_set in test_sets.items():
        print(set_name+":")
        save_folder = os.path.join('./result/crf', set_name)
        print(save_folder)
        check_path(save_folder)
       
        titer = test_set.size
 
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            out_shape = gt.shape
    
            pred = gt

            mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
            std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

            image = nn.functional.interpolate(image, size=out_shape, mode='bilinear')

            orig_img = image[0].numpy().transpose(1, 2, 0)
            orig_img = ((orig_img * std + mean) * 255.).astype(np.uint8)
                
            pred = (pred > 0.5).astype(np.uint8)
            pred = crf_inference_label(orig_img, pred)
            pred = cv2.medianBlur(pred.astype(np.uint8), 7)
             
            #for i in range(len(name)-1,0,-1):
            #    if(name[i]=="\\"):
            #        name = name[i+1:]
            #        break

            im_path = os.path.join(save_folder, name + '.png')
            Image.fromarray((pred * 255)).convert('L').save(im_path)
                         
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
       


def main():
  
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        print('Need dataset name!')
        return

    net_name = 'cornet_compare'   
    config, schedule = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    config['only_crf'] = True

    saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
   
    test_sets = OrderedDict()  
    config['vals']=[dataset_name]
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    test_model(test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()
