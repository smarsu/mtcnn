# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

"""Crop the face"""
import os 
import os.path as osp
import cv2
import numpy as np

def crop(cvimg, box, path=None):
    """Crop the face box from the cvimg.
    
    Args:  
        cvimg: ndarray, 
        box: list of int, [x1, y1, x2, y2]
        path: str, if True, save the croped face to path.

    Returns:
        croped_img: ndarray
    """
    img_h, img_w, _ = cvimg.shape
    
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1

    zeros = np.zeros(shape=(h, w, 3))
    left = -x1 if x1 < 0 else 0
    top = -y1 if y1 < 0 else 0
    right = -(x2 - img_w) if x2 > img_w else w
    bottom = -(y2 - img_h) if y2 > img_h else h

    x1 = min(max(x1, 0), img_w)
    y1 = min(max(y1, 0), img_h)
    x2 = max(min(x2, img_w), 0)
    y2 = max(min(y2, img_h), 0)

    zeros[top:bottom, left:right] = cvimg[y1:y2, x1:x2] 
    croped_img = zeros

    if path:
        dir = osp.split(path)[0]
        if not osp.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(path, croped_img)
    return croped_img
