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



from base.framework_factory import load_framework

def test_model(model, test_sets, config, epoch=None, saver=None):
    model.eval()
    
    st = time.time()
    for set_name, test_set in test_sets.items():
        print('###############')
        print(set_name+':')
        st = time.time()
        time_count = 0

        titer = test_set.size

        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):

            image, gt, name = test_set.load_data(j)
            out_shape = gt.shape
            tmp = time.time()
            Y = model(image.cuda())
            
            ###continue

            
            if config['crf']:
                Y['final'] = nn.functional.interpolate(Y['final'], size=out_shape, mode='bilinear')          
                pred = Y['final'].sigmoid_().cpu().data.numpy()[0, 0]
                pred, gt = normalize_pil(pred, gt)
                mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
                std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

                ### to keep the same shape
                image = nn.functional.interpolate(image, size=out_shape, mode='bilinear')

                orig_img = image[0].numpy().transpose(1, 2, 0)
                orig_img = ((orig_img * std + mean) * 255.).astype(np.uint8)
                
                pred = (pred > 0.5).astype(np.uint8)
                pred = crf_inference_label(orig_img, pred)
                pred = cv2.medianBlur(pred.astype(np.uint8), 7)
           
            time_count += time.time() - tmp
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        print('.')
        print(time_count)
        #print('Test using time: {}.'.format(round(time.time() - st, 3)))


def main():
    print("######test_speed:")
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    if config['crf']:
        config['orig_size'] = True
    
    if config['weight'] != '':
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    else:
        print('No weight file provide!')


    test_sets = OrderedDict()
    
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    model = model.cuda()
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()
