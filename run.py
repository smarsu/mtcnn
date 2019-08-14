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
from mtcnn import PNet, RNet, ONet
from square import square_boxes
from broad import broad_boxes
from crop import crop

np.random.seed(196)

dataset_root = '/datasets/wider'


if __name__ == '__main__':
    parse = sys.argv[1] #  'train'
    net = sys.argv[2]
    #net = 'pnet'

    if parse == 'train':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split/wider_face_train_bbx_gt.txt')

        if net == 'pnet':
            pnet = PNet(batch_size=32, no_mask=False, rd_size=False, min_face=60, scale_factor=0.89)
            # 0.8780253_1_0.1_pnet.npz the last size of step 1 for lr 0.1.
            pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet.npz'))
            pnet.train(widerface.train_datas, 100, lr=0.001)
            # pnet.train(widerface.train_datas_debug, 100, lr=0.1)
            #conf, box = pnet.test(widerface.train_datas_debug(1)[0][0][0])
            #print(conf)
            #print(box)
            #conf, box = pnet._check_model_capable(widerface.train_datas_debug)
            #print(conf)
            #print(box)
        elif net == 'rnet':
            widerfacepnet = WiderFace('/datasets/wider/images', 
                '/datasets/wider/wider_face_split/wider_face_train_pbbx_gt.txt')
            pnet = PNet()
            rnet = RNet(scale_mask=False, batch_size=2)
            rnet.sess.restore(osp.join(pnet.model_root, '0.06339169_7_0.001_rnet.npz'))
            rnet.train(widerfacepnet.train_datas, 100, 0.001, widerface.data_map, weight_decay=0)
        elif net == 'onet':
            pwiderface = WiderFace('/datasets/wider/images', 
                '/datasets/wider/wider_face_split/wider_face_train_pbbx_gt.txt')
            rwiderface = WiderFace('/datasets/wider/images', 
                '/datasets/wider/wider_face_split/wider_face_train_rbbx_gt.txt')
            rwiderface.merge(pwiderface)
            pnet = PNet()
            rnet = RNet()
            onet = ONet(scale_mask=False, batch_size=2)
            #onet.sess.restore(osp.join(onet.model_root, '0.5085983_29_0.1_onet.npz'))
            onet.train(rwiderface.train_datas, 100, 0.01, widerface.data_map, weight_decay=0., stage='onet')
        else:
            raise NotImplementedError

    elif parse == 'check':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split/wider_face_train_bbx_gt.txt')

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
                              '/datasets/wider/wider_face_split/wider_face_train_bbx_gt.txt')
        
        if net == 'pnet':
            pnet = PNet(scale_factor=0.89, conf_thrs=0.95, nms_thrs=0.1, min_face=24, nms_topk=32)
            pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet.npz'))
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

        elif net == 'rnet':
            pwiderface = WiderFace('/datasets/wider/images', 
                                   '/datasets/wider/wider_face_split/wider_face_train_pbbx_gt.txt')
            pnet = PNet()
            rnet = RNet(conf_thrs=0.3)
            rnet.sess.restore(osp.join(rnet.model_root, '0.022445953_53_0.001_rnet.npz'))
            with open(osp.join(dataset_root, 
                               'wider_face_split', 
                               'wider_face_train_rbbx_gt.txt'), 'w') as fb:
                for images, boxes in tqdm(pwiderface.train_datas(1)):
                    image = images[0]
                    raw_image = cv2.imread(image)
                    conf, box = rnet.test(raw_image, boxes[0])
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
                              '/datasets/wider/wider_face_split/wider_face_train_bbx_gt.txt')

        if net == 'pnet':
            image = widerface.train_datas_debug(32)[0][0][2]
            #image = 'data/demo/face.jpg'
            pnet = PNet(scale_factor=0.89, conf_thrs=0.95, nms_thrs=0.1, min_face=24, nms_topk=32)
            pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet.npz'))
            conf, box = pnet.test(image)
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = cv2.imread(image)
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'face_result.jpg']), image)
        elif net == 'rnet':
            image = widerface.train_datas_debug(32)[0][0][0]
            image = 'data/demo/face.jpg'
            pnet = PNet(scale_factor=0.89, conf_thrs=0.96, nms_thrs=0.1, min_face=60, nms_topk=None)
            pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet_v2.npz'))
            conf, box = pnet.test(image)
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            raw_image = cv2.imread(image)
            image = raw_image.copy()
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'p_face_result.jpg']), image)

            rnet = RNet(conf_thrs=0.5)
            rnet.sess.restore(osp.join(rnet.model_root, '0.022445953_53_0.001_rnet.npz'))
            conf, box = rnet.test(raw_image, bboxs)
            print('RNet result:')
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = raw_image.copy()
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'r_face_result.jpg']), image)
        elif net == 'onet':
            image = widerface.train_datas_debug(32)[0][0][6]
            image = 'data/demo/ffy1.jpg'
            pnet = PNet(scale_factor=0.89, conf_thrs=0.8, nms_thrs=0.5, min_face=60, nms_topk=32)
            pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet_v2.npz'))
            conf, box = pnet.test(image)
            print('PNet result:')
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            raw_image = cv2.imread(image)
            image = raw_image.copy()
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'p_face_result.jpg']), image)

            rnet = RNet(conf_thrs=0.5)
            rnet.sess.restore(osp.join(rnet.model_root, '0.022445953_53_0.001_rnet.npz'))
            conf, box = rnet.test(raw_image, bboxs)
            print('RNet result:')
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = raw_image.copy()
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'r_face_result.jpg']), image)

            onet = ONet(conf_thrs=0.5)
            onet.sess.restore(osp.join(onet.model_root, '0.012311436_69_0.01_onet.npz'))
            conf, box = onet.test(raw_image, bboxs)
            print('ONet result:')
            print(conf)
            print(box)
            bboxs = box.astype(np.int32)
            image = raw_image.copy()
            for x1, y1, x2, y2 in bboxs:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imwrite('/'.join([pnet.demo_root, 'o_face_result.jpg']), image)

            bboxs = square_boxes(bboxs)
            bboxs = broad_boxes(bboxs, 0.4)
            for idx, box in enumerate(bboxs):
                crop(raw_image.copy(), box, str(idx) + '.jpg')
    else:
        raise ValueError('Unsupported argv parse {}, expect '
                         '[train, check, test]'.format(parse))
