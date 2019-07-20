# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
import sys
from tqdm import tqdm
import cv2
import numpy as np

from datasets import WiderFace
from mtcnn import PNet
from square import square_boxes

np.random.seed(196)

dataset_root = '/datasets/wider'


if __name__ == '__main__':
    parse = sys.argv[1] #  'train'
    net = 'pnet'

    if parse == 'train':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            pnet = PNet(batch_size=32, no_mask=True, rd_size=False)
            # 0.8780253_1_0.1_pnet.npz the last size of step 1 for lr 0.1.
            pnet.sess.restore(osp.join(pnet.model_root, '23.400742_rand_size_8_0.01_pnet.npz'))
            pnet.train(widerface.train_datas, 100, lr=0.01)
            # pnet.train(widerface.train_datas_debug, 100, lr=0.1)
            conf, box = pnet.test(widerface.train_datas_debug(1)[0][0][0])
            print(conf)
            print(box)
            #conf, box = pnet._check_model_capable(widerface.train_datas_debug)
            #print(conf)
            #print(box)
        else:
            raise NotImplementedError

    elif parse == 'check':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            image = widerface.train_datas_debug(1)[0][0][0]
            pnet = PNet()
            conf, box = pnet._check_model_capable(widerface.train_datas_debug)
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = cv2.imread(image)
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2),(0,255,0),2)
            cv2.imwrite('/'.join([pnet.demo_root, 'face_result.jpg']), image)
        else:
            raise NotImplementedError

    elif parse == 'gene':
        """Generator the dataset for next stage."""
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')
        
        if net == 'pnet':
            pnet = PNet(conf_thrs=0.1)
            pnet.sess.restore(osp.join(pnet.model_root, '100.8965_(203, 203)_7_0.0001_pnet.npz'))
            with open(osp.join(dataset_root, 
                               'wider_face_split', 
                               'wider_face_train_pbbx_gt.txt'), 'w') as fb:
                for images, _ in tqdm(widerface.train_datas(1)):
                    image = images[0]
                    conf, box = pnet.test(image)
                    box = square_boxes(box).astype(np.int32)
                    for bbox in box:
                        x1, y1, x2, y2 = bbox
                        h = y2 - y1
                        w = x2 - x1

                        fb.write(image + '\n')
                        fb.write('1' + '\n')  # 1 pic
                        fb.write(('{} ' * 10).format(x1, y1, w, h, 0, 0, 0, 0, 0, 0) + '\n')
        else:
            raise NotImplementedError

    elif parse == 'test':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            image = widerface.train_datas_debug(32)[0][0][0]
            #image = 'data/demo/face.jpg'
            pnet = PNet(conf_thrs=0.01, nms_thrs=0.3, min_face=60)
            pnet.sess.restore(osp.join(pnet.model_root, '65.45905_rand_size_0_0.01_pnet.npz'))
            conf, box = pnet.test(image)
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = cv2.imread(image)
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'face_result.jpg']), image)
    else:
        raise ValueError('Unsupported argv parse {}, expect '
                         '[train, check, test]'.format(parse))
