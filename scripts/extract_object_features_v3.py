#!/usr/bin/env python
'''
After you have extracted object features using botoom-up and top-down attention Resnet
(see https://github.com/peteanderson80/bottom-up-attention), you can use this script to
extract visual features from Flickr30K as OVC did.

Requirements: python 2.7 
'''
import csv
import sys
import base64
import numpy as np
from timer import Timer
import json
import cv2
csv.field_size_limit(sys.maxsize)

## make sure these data addrs right in your case
multi30k_entity_dataset_address = '~/MMT/multi30k-entities-dataset'
bottom_up_attention_address = '~/bottom-up-attention'
multi30K_German_address = 'data/Multi30K_DE'

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic


def read_bottom_up_features(tsv_file):
    objects_features = []
    image_ids = []
    with open(tsv_file, "r+b") as tsvfile:
        FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
        reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id']  = int(item['image_id'])
            item['image_h']   = int(item['image_h'])
            item['image_w']   = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            item['features']  = np.frombuffer(base64.decodestring(item['features']), dtype=np.float32).reshape((item['num_boxes'], -1))
            if item['image_id'] not in image_ids:
                image_ids.append(item['image_id'])
                objects_features.append((item['image_id'], item['features']))
            else: ## if repreated
                print(objects_features[item['image_id']] - item['features'])
    ## sort the features in the order of its ordinary dataset
    objects_features = sorted(objects_features, key=lambda x:x[0])
    return objects_features



if __name__ == '__main__':
    # ############################################################################################
    # ## (Duplicated in OVC) rough multi30k 
    # splits = ['val', 'train', 'test']
    # img_num = {
    #     'val':  1014,
    #     'test': 1000,
    #     'train':29000
    # }
    # T = Timer()
    # for split in splits:
    #     # get feaures only
    #     tsv_file = bottom_up_attention_address + '/%s_flickr30K_merged.tsv' % split
    #     features = read_bottom_up_features(tsv_file)
    #     assert len(features) == img_num[split]
    #     features = np.array([fea[1] for fea in features])
    #     print(features.shape)
    #     ## save
    #     T.tic()
    #     np.save(multi30K_German_address + '/%s.norm.tok.lc.10000bpe_ims_bta.npy' % split, features)
    #     T.toc()
    #     print('============= spent', T.average_time/3600., 's', 'on', split)










    ############################################################################################
    ## detailed multi30k
    splits = ['val', 'test']
    img_num = {
        'val':  1014,
        'test': 1000,
    }
    T = Timer()
    for split in splits:
        T.tic()

        image_lists_file = multi30K_German_address + '/%s_images.txt' % split
        with open(image_lists_file, 'r') as f: image_lists = [l.strip() for l in f.readlines()]

        json_file = 'data/flickr30K-' + split + '.json'
        with open(json_file, 'r') as f: items = [json.loads(s) for s in f]
        
        ## resort the json in the original order of multi30k
        new_json_file = 'data/' + split + '.json'
        with open(new_json_file, 'w') as new_f:
            new_items = []
            for img in image_lists:
                for i, it in enumerate(items):
                    if it['image_name'].split('/')[-1] == img:
                        json.dump(it, new_f)
                        new_f.write('\n')
                        new_items.append(it)
                        break
            items = new_items
            print(len(items))

        image_name, features, obj_confs, boxes, obj_categories = [], [], [], [], []
        max_len = 0

        for i, it in enumerate(items):
            # [u'obj_boxes', u'features', u'obj_confs', u'image_name', u'boxes', u'obj_categories']
            assert it['image_name'].split('/')[-1] == image_lists[i], (it['image_name'], image_lists[i])

            ## debug conf_thresh
            # conf_thresh = 0.48
            # reliable_order = np.where(np.array(it['obj_confs']) >= conf_thresh)[0]
            # print(reliable_order)
            # exit()

            ## selected by the objects' area to clear objects whose area is unreasonably large
            obj_boxes = np.array(it['obj_boxes'])
            img = multi30k_entity_dataset_address \
                    +'/data/images/flickr30k-images/'\
                    + ('task1/' if split == 'test' else '') \
                    + image_lists[i]
            img = cv2.imread(img)
            obj_areas = (obj_boxes[:,2] - obj_boxes[:,0]) * (obj_boxes[:,3] - obj_boxes[:,1])
            total_img_area = float(img.shape[0] * img.shape[1])
            obj_area_rate = (obj_areas / total_img_area)
            reliable_order = np.where(obj_area_rate <= np.max((0.75, np.min(obj_area_rate)), axis=-1) )[0]
            print(reliable_order)

            max_len = max(max_len, reliable_order.shape[0])
            image_name.append(it['image_name'])
            features.append(np.array(it['features'])[reliable_order])
            obj_confs.append(np.array(it['obj_confs'])[reliable_order])
            boxes.append(np.array(it['obj_boxes'])[reliable_order])
            obj_categories.append(np.array(it['obj_categories'])[reliable_order])
            assert len(obj_categories[-1]) == features[-1].shape[0], \
                        (len(obj_categories[-1]), features[-1].shape[0])
        assert len(image_lists) == len(image_name)

        ## padding
        for i, fea in enumerate(features): 
            features[i] = np.concatenate((fea, np.zeros((max_len-fea.shape[0], fea.shape[1]))), axis=0)
        features = np.array(features)
        print len(features), 'images', 'max number of objects in an image =', max_len
        assert len(features) == img_num[split]

        ## save
        np.save(multi30K_German_address + '/%s.norm.tok.lc.10000bpe_ims_bta_sort.npy' % split, features)
        # all object lists for future visualization
        with open(multi30K_German_address + '/%s_obj_list.json' % split, 'w') as f:
            for obj_category, box in zip(obj_categories, boxes):
                itm = {}
                itm['obj_category'] = obj_category.tolist()
                itm['box'] = box.tolist()
                json.dump(itm, f)
                f.write('\n')
        T.toc()
        print '============= spent', T.average_time/3600., 's', 'on', split






    ############################################################################################
    ## resorted the multi30k according to object-word similarity for training OVC
    ## note that all test data did not need the processing.
    splits = ['val', 'train']
    T = Timer()
    for split in splits:
        T.tic()

        image_lists_file = multi30K_German_address + '/%s_images.txt' % split
        with open(image_lists_file, 'r') as f: image_lists = [l.strip() for l in f.readlines()]
        print image_lists_file, 'loading done.'

        sentence_file = multi30K_German_address + '/%s.norm.tok.lc.en' % split
        with open(sentence_file, 'r') as f: sentences = [l.strip() for l in f.readlines()]
        print sentence_file, 'loading done.'

        json_file = 'data/' + split + '.json'
        with open(json_file, 'r') as f: items = [json.loads(s) for s in f]
        print json_file, 'loading done.'

        detection_sim_file = 'data/%s_obj_similarity' % split
        with open(detection_sim_file, 'r') as f: obj_sims = [l.split() for l in f.readlines()]
        print detection_sim_file, 'loading done.'

        image_names, features, obj_confs, boxes, obj_categories = [], [], [], [], []
        max_len = 0

        for i, (it, sent) in enumerate(zip(items, sentences)):
            sent = [s.replace('people', 'person').replace('woman', 'person').replace('man', 'person') \
                        for s in sent.split()] ## merge person-relevant words into person class.
            assert it['image_name'].split('/')[-1] == image_lists[i], (it['image_name'], image_lists[i])

            for ii, obj in enumerate(it['obj_categories']):
                obj = str(obj).split()[-1]
                obj = obj.replace('people', 'person').replace('woman', 'person').replace('man', 'person')
                it['obj_categories'][ii] = obj

            obj_sim = np.array([float(sim) for sim in obj_sims[i]])
            for ii, obj in enumerate(it['obj_categories']):
                obj = str(obj).split()[-1]
                if obj in sent: obj_sim[ii] = 1.

            ## similarity reorder type
            # 1. type 1, used in OVC
            order = np.argsort(obj_sim)[::-1] # resort the score by similarity value
            obj_sim = obj_sim[order]
            reliable_type = obj_sim[:, None]

            # # 2. type 2, duplicated in OVC
            # conf_thresh = 0.5
            # reliable_order = np.where(obj_sim >= conf_thresh)[0]
            # reliable_type = np.array([1.] * reliable_order.shape[0] + \
            #                 [0.] * (obj_sim.shape[0] - reliable_order.shape[0]))[:,None]

            ## get features of the single image
            obj_conf = np.array(it['obj_confs'])[order][:, None]
            box = np.array(it['obj_boxes'])[np.array(np.array(it['boxes']))[order]]
            obj_category = np.array(it['obj_categories'])[order]
            image_name = it['image_name']
            feature = np.array(it['features'])[order] # np.concatenate((feature, box, obj_conf, reliable_type), axis=-1)

            ## get reorder features of the single image
            reduplicate_order = []
            reduplicate_obj_category = []
            for ii, obj in enumerate(obj_category.tolist()):
                if obj not in reduplicate_obj_category:
                    reduplicate_obj_category.append(obj)
                    reduplicate_order.append(ii)
            reduplicate_order = np.array(reduplicate_order)
            obj_conf = obj_conf[reduplicate_order]
            box = box[reduplicate_order]
            obj_category = obj_category[reduplicate_order]
            feature = feature[reduplicate_order]
            obj_sim = obj_sim[reduplicate_order]

            ##
            max_len = max(max_len, obj_conf.shape[0])
            image_names.append(image_name)
            features.append(feature)
            obj_confs.append(obj_conf)
            boxes.append(box)
            obj_categories.append(obj_category)
            assert len(obj_categories[-1]) == features[-1].shape[0], (len(obj_categories[-1]), features[-1].shape[0])

            if i % 1000 == 0:
                print max_len, image_name, '\n', obj_sim, '\n', obj_category, '\n', sent, '\n'	
            
        assert len(image_lists) == len(image_names)

        ## saving now
        for i, fea in enumerate(features): # padding
            if fea.shape[0] != max_len:
                features[i] = np.concatenate((fea, np.zeros((max_len-fea.shape[0], fea.shape[1]))), axis=0)
        assert len(features) == img_num[split]
        print len(features), 'images', 'max number of objects in an image =', max_len
        print 'saving to npy...'
        np.save(multi30K_German_address + '/%s.norm.tok.lc.10000bpe_ims_bta_sort_with_original_similarity.npy' % split, features)

        ## saving all object lists
        with open(multi30K_German_address + '/%s_obj_list_with_original_similarity.json' % split, 'w') as f:
            for obj_category, box in zip(obj_categories, boxes):
                itm = {}
                itm['obj_category'] = '||'.join(obj_category.tolist())
                itm['box'] = box.tolist()
                json.dump(itm, f)
                f.write('\n')
        T.toc()
        print '============= spent', T.average_time / 3600., 's', 'on', split
    
    print 'extraction done.'