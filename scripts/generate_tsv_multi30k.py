#!/usr/bin/env python

"""
Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
separate tsv file that can be merged later (e.g. by using merge_tsv function). 
Modify the load_image_ids script as necessary for your data location.
"""
## Update for VCR
## How to generate VCR bottom-up features?
# 1. set down the dependencies: python 2.7 caffe
# 2. then replace the generate_tsv.py in BOTTOM-UP-ROOT/tools/generate_tsv.py
# 3. check the vcr image_list file path and then run ". extract_vcr_feature.sh"


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
import math

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 36
MAX_BOXES = 36

gpu_num = 1


def load_flickr30k_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    assert split_name in ['flickr30K-train', 'flickr30K-val', 'flickr30K-test']
    split_name = split_name.split('-')[-1]
    IMAGES_DIR = '/home/dxwang/code/MMT/multi30k-entities-dataset/data/images/flickr30k-images'
    IMAGES_DIR += '/task1' if split_name == 'test' else ''
    ANNOTS_DIR = os.path.join('/home/dxwang/code/MMT/A-Visual-Attention-Grounding-Neural-Model/data/Multi30K_DE/')
    split = []  

    with open(os.path.join(ANNOTS_DIR, '%s_images.txt' % split_name), 'r') as f:
        items = [s.strip() for s in f.readlines()]
        for i, item in enumerate(items):
            filepath = os.path.join(IMAGES_DIR, item)
            split.append((filepath, i))      
    return split

def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    if im is None: print(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }   


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='flickr30K-train, flickr30K-val, flickr30K-test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    
def generate_tsv(gpu_id, prototxt, weights, missing, outfile ):
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
            _t = {'misc' : Timer()}
            count = 0

            for im_file, image_id in missing:
                _t['misc'].tic()
                writer.writerow(get_detections_from_im(net, im_file, image_id))
                _t['misc'].toc()
                if (count % 100) == 0:
                    print 'GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                          .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
                          _t['misc'].average_time*(len(missing)-count)/3600)
                count += 1



def merge_tsvs(split, gpu_list):
    img_num = {
        'val':  1014,
        'test': 1000,
        'train':29000
    }
    img_num = img_num[split]
    test = ['/home/dxwang/bottom-up-attention/test2014_resnet101_faster_rcnn_flickr30K_%s.tsv.%s' % (split, gpu_list[i]) for i in range(gpu_num)]

    image_ids = np.ones(img_num)
    outfile = '/home/dxwang/bottom-up-attention/%s_flickr30K_merged.tsv' % split

    ## exists in raw file
    if os.path.exists(outfile):
        with open(outfile, 'r+b') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                image_ids[int(item['image_id'])] = 0


    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)          
        for infile in test:
            if os.path.exists(infile):
                with open(infile, "r+b") as tsv_in_file:
                    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                    for item in reader:
                        if image_ids[int(item['image_id'])] == 0:
                            continue
                        #print(item['num_boxes'])
                        try:
                            writer.writerow(item)
                            image_ids[int(item['image_id'])] = 0
                        except Exception as e:
                            print e, '\n', item['image_id']

                            # return False
    if np.sum(image_ids) != 0:
        for i in range(img_num):
            if image_ids[i] !=0:
                print i,
        return False
    else:
        print 'done'
        return True
                      

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    if '0' in gpu_list and len(gpu_list) != 1:
        gpu_list.remove('0') # gpu-0 always be occupied when running this script for some unknown reason (2080ti)
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_flickr30k_image_ids(args.data_split)
    caffe.init_log()

    ## 
    found_ids   = set()
    wanted_ids  = set([int(image_id[1]) for image_id in image_ids])
    for gpu_id in gpu_list:
        outfile = '%s.%s' % (args.outfile, gpu_id)
        if os.path.exists(outfile):
            with open(outfile) as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    found_ids.add(int(item['image_id']))
    wanted_ids = wanted_ids - found_ids

    # wanted_ids = set([23036,34526,68993])

    if len(wanted_ids) == 0:
        if merge_tsvs(args.data_split.split('-')[-1], gpu_list):
            print('done')
            exit()

    missing = [ image_ids[i] for i in wanted_ids]

    # caffe.log('Using devices %s. Missing %d images' % (str(gpus), len(missing)))
    print('Using devices %s. Missing %d images' % (str(gpus), len(missing)))

    procs = []    
    average_num =  int(math.ceil(float(len(wanted_ids)) / float(len(gpus)) ))

    for i, gpu_id in enumerate(gpus):
        if i*average_num > len(wanted_ids):
            break

        outfile = '%s.%d' % (args.outfile, gpu_id)

        p = Process(target=generate_tsv,
                    args=(
                        gpu_id,
                        args.prototxt,
                        args.caffemodel,
                        missing[min(i*average_num, len(wanted_ids)):min((i+1)*average_num, len(wanted_ids))],
                        outfile
                    )
                )
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()  
        
    merge_tsvs(args.data_split.split('-')[-1], gpu_list)
          
