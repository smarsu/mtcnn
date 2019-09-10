# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp

import cv2
import numpy as np

from mtcnn import PNet, RNet, ONet
from square import square_boxes
from broad import broad_boxes
from crop import crop

pnet = PNet(scale_factor=0.89, conf_thrs=0.8, nms_thrs=0.5, min_face=60, nms_topk=32)
pnet.sess.restore(osp.join(pnet.model_root, '3.2153504_cycle_7_0.01_pnet_v2.npz'))

rnet = RNet(conf_thrs=0.5)
rnet.sess.restore(osp.join(rnet.model_root, '0.022445953_53_0.001_rnet.npz'))

onet = ONet(conf_thrs=0.5)
onet.sess.restore(osp.join(onet.model_root, '0.012311436_69_0.01_onet.npz'))


def detect(img, top_k=-1):
    """Do face detection with the input img.
    
    Args:
        img: ndarray, shape with [h, w, 3]
        top_k: output with the top k detected faces. default for -1, which 
            means return all detected bboxes.

    Returns:
        bboxes: ndarray, [n, 4]
    """
    h, w, c = img.shape
    if c != 3:
        print('WARNING: wrong input shape {}, we need the c to be 3.'.format(c))
        return np.array([])

    confs, bboxes = pnet.test(img)
    bboxes = np.round(bboxes).astype(np.int32)
    confs, bboxes = rnet.test(img, bboxes)
    bboxes = np.round(bboxes).astype(np.int32)
    confs, bboxes = onet.test(img, bboxes)
    bboxes = np.round(bboxes).astype(np.int32)

    if len(confs) != 1:
        print('WARNING: Unexpected face num ', len(confs))

    sorted_id = np.argsort(-confs.flatten())[:top_k]
    keeped_bboxes = bboxes[sorted_id]
    return keeped_bboxes


def crop_face(img, bboxes, dst_face_path):
    """Crop face with the raw image and face boxes.
    
    Args:
        img: ndarray, shape with [h, w, 3]
        bboxes: ndarray, [n, 4]
        dst_face_path: str, the path to write the face.
    """
    h, w, c = img.shape
    n, axis = bboxes.shape
    if c != 3:
        print('ERROR: wrong input shape {}, we need the c to be 3.'.format(c))
        exit()
    if axis != 4:
        print('ERROR: wrong input shape {}, we need the axis to be 4.'.format(axis))
        exit()

    bboxes = np.round(bboxes).astype(np.int32)
    bboxes = square_boxes(bboxes)
    bboxes = broad_boxes(bboxes, 0.4)
    for idx, bbox in enumerate(bboxes):
        crop(img, bbox, dst_face_path)


def main():
    lfw_root = '/datasets/lfw'
    lfw_detected_root = '/datasets/lfw_detected'
    lfw_person_path = 'lfw_person.txt'
    with open(lfw_person_path, 'r') as fb:
        lines = fb.readlines()
        persons_path = [osp.join(lfw_root, line.strip()) for line in lines]
        dst_persons_path = [osp.join(lfw_detected_root, line.strip()) 
                            for line in lines]

    for person_path, dst_person_path in zip(persons_path, dst_persons_path):
        print('INFO: run ', person_path, '...')
        img = cv2.imread(person_path)
        if img is None:
            print('ERROR: read {} failed'.format(person_path))
            exit()
        bboxes = detect(img, top_k=1)
        crop_face(img, bboxes, dst_person_path)


if __name__ == '__main__':
    main()
