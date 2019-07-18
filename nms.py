# --------------------------------------------------------
# NMS
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np
from overlap import bbox_overlap


def nms(boxes, thresh):
    """boxes will be sorted in this function

    Args:
        boxes: Tensor, [n, 5], [x1, y1, x2, y2, score, type]
    Return:
        rtboxes: Tensor, [k, 5], the satisify box.
    """
    order = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while order.size > 0:
        keep.append(order[0])
        overlaps = bbox_overlap(boxes[order[0:1]][:, :4], 
                                boxes[order[1:]][:, :4]).flatten()

        ids = np.where(overlaps<thresh)[0]
        order = order[ids + 1]
    
    return keep


if __name__ == '__main__':
    boxes = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [2, 4, 5, 6, 0.7], [1, 2, 7, 6, 0.8]])
    print(nms(boxes, 0.2))
