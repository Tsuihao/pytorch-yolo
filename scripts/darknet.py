import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 

def parse_cfg(config_file):
    '''
    parse the configuration file
    '''
    config = open(config_file, 'r')
    file = config.read().split('\n')
    
    # clean up
    file = [line for line in file if len(line) > 0 and line[0]!= '#']  
    file = [line.lstrip().rstrip() for line in file]
    
    # store the result in a list
    network_blocks = []
    block = {}
    
    for l in file:
        if l[0] == '[':
            if len(block) != 0:  # block contain previos layer info
                network_blocks.append(block)
                block = {}
            block['type'] = l[1:-1].rstrip() # get the name inside ['name']
        else:
            entity, value = l.split('=')
            block[entity.rstrip()] = value.lstrip()
        
    network_blocks.append(block) # last one
    
    return network_blocks


class dummy_layer(nn.Module):
    def __init__(self):
        super(dummy_layer, self).__init__()
    
class detector(nn.Module):
    def __init__(self, anchors):
        super(detector, self).__init__()
        self.anchors = anchors


def construct_darknet(network_blocks):
    '''
    using nn.Sequential()
    '''
    darknet_info = network_blocks[0]
    modules = nn.ModuleList([])
    channels = 3
    filterTracker = []

    for i, x in enumerate(network_blocks[1:]):  # i = index, x = each dicts
        seq_modoule = nn.Sequential()
        
        # process base on the layer type
        if (x['type'] == 'convolutional'):
            filters = int(x['filter'])
            is_pad  = int(x['pad'])
            kernel  = int(x['size'])
            stride  = int(x['stride'])

            if is_pad:
                padding = (kernel - 1) // 2
            else:
                padding = 0
            
            activation = x['activation']

            # specially handling for the only 1 convolution-layer without bn
            try : 
                bn = int(x['batch_normalize'])
                bias = False
            except:
                bn = 0
                bias = True
            conv = nn.Conv2d(channels, filters, kernel, stride, padding, bias=bias)
            seq_modoule.add_module('conv_{}'.format(i), conv)

            if bn:
                bn = nn.BatchNorm2d(filters)
                seq_modoule.add_module('batch_norm_{}'.format(i), bn)
            
            if activation == 'leaky':
                activation = nn.LeakyReLU(0.1, inplcae=True)
                seq_module.add_module('leaky_{}'.format(i), activation)
        
        elif (x['type'] == 'upsample'):
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            seq_modoule.add_module('upsample_{}'.format(i), upsample)

        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')  # e.g {'type': 'route', 'layers': '-1, 61'}, {'type': 'route', 'layers': '-4'}
            start = int(x['layers'][0])
            try: # check the e.g. sometime, there is no end
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i

            route = dummy_layer()
            seq_modoule.add_module('route_{}'.format(i), route)

            # what is this?
            if end < 0:
                filters = filterTracker[i+start] + filterTracker[i+end]
            else:
                filters = filterTracker[i+start]
            
        elif (x['type'] == 'shortcut'):
            shortcut = dummy_layer()
            seq_modoule.add_module('shortcut_{}'.format(i), shortcut)

        elif (x['type'] == 'yolo'):
            