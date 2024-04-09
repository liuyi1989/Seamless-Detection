import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1) ###
    yield nn.BatchNorm2d(cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')

def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1) ###
    yield nn.BatchNorm2d(cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.BatchNorm2d(cout)
    yield nn.ReLU(inplace=True)



class RAM_(nn.Module):
    def __init__(self, tar_feat):
        super(RAM_, self).__init__()

        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        self.res_conv1 = nn.Conv2d(tar_feat, tar_feat, 1, padding=0)
        self.res_conv2 = nn.Conv2d(tar_feat, tar_feat, 1, padding=0)
        self.fuse = nn.Conv2d(tar_feat * 3, tar_feat, 3, padding=1)

    def forward(self, xs0, xs1, glob_x):
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=xs0.size()[2:], mode=mode)
        loc_x1 = xs0
        loc_x2 = nn.functional.interpolate(xs1, size=xs0.size()[2:], mode=mode)
        loc_x = self.fuse(torch.cat([loc_x1, loc_x2 , glob_x0], dim=1))
        return loc_x




class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        #self.adapter = [nn.Sequential(*list(up_conv(feat[i], feat[0], False))).cuda() for i in range(5)]
        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0], False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0], False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0], False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0], False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0], False)))
        

        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))


        self.ram_0 = RAM_(feat[0])
        self.ram_1 = RAM_(feat[0])

        self.fc_bg = nn.Linear(40 * 40, 64)
        self.norm_bg = nn.BatchNorm2d(1)

        self.fc_fg  = nn.Linear(40 * 40, 64)
        self.norm_fg = nn.BatchNorm2d(1)
        
    def forward(self, xs, y, x_size, phase = 'test'):


        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])

        

        glob_x = xs[4]
        reg_x = self.ram_0(xs[1], xs[3], glob_x)
        glob_x = self.gb_conv(glob_x)
        loc_x = self.ram_1(xs[0], xs[2], glob_x)
        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')


        if(phase =='train'):
            fg = nn.functional.interpolate(pred, size=xs[2].size()[2:], mode='bilinear')
            fg = self.norm_fg(fg)
            fg = torch.flatten(fg, 1)
            fg_v = self.fc_fg(fg)
        

            bg = xs[2].detach()
            mask = y.detach()
            mask = 1 - mask
            mask = nn.functional.interpolate(mask, size=xs[2].size()[2:], mode='bilinear')
            bg = mask * bg
            bg = torch.sum(bg, dim=1, keepdim=True)
            bg = self.norm_bg(bg)
            bg = torch.flatten(bg, 1)
            bg_v = self.fc_bg(bg)
            
        OutDict = {}
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred
        if(phase =='train'):
            OutDict['fg_v'] = fg_v
            OutDict['bg_v'] = bg_v

        return OutDict
        

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, y = 0, phase='test'):

        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, y, x_size,phase)
        return out
