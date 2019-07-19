# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
import sys
import cv2
import numpy as np
from datasets import WiderFace
from mtcnn import PNet

np.random.seed(196)


if __name__ == '__main__':
    parse = sys.argv[1] #  'train'
    net = 'pnet'

    if parse == 'train':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            pnet = PNet()
            # 0.8780253_1_0.1_pnet.npz the last size of step 1 for lr 0.1.
            pnet.sess.restore(osp.join(pnet.model_root, '0.8780253_1_0.1_pnet.npz'))
            pnet.train(widerface.train_datas, 10, lr=0.01)
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

    elif parse == 'test':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            image = widerface.train_datas_debug(1)[0][0][0]
            pnet = PNet(conf_thrs=0.9)
            pnet.sess.restore(osp.join(pnet.model_root, '0.67729115_(12, 12)_1_0.01_pnet.npz'))
            conf, box = pnet.test(widerface.train_datas_debug(1)[0][0][0])
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = cv2.imread(image)
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2),(0,255,0),2)
            cv2.imwrite('/'.join([pnet.demo_root, 'face_result.jpg']), image)
    else:
        raise ValueError('Unsupported argv parse {}, expect '
                         '[train, check, test]'.format(parse))
