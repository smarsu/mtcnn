# --------------------------------------------------------
# broad box
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np


def broad_boxes(boxes, factor=0.2):
    """
    Args:
        boxes: ndarray, [n, 4], (x1, y1, x2, y2)

    Returns:
        boxes: ndarray
    """
    boxes = boxes.astype(np.float32)
    # [n, 1]
    h, w = boxes[:, 3:4] - boxes[:, 1:2], boxes[:, 2:3] - boxes[:, 0:1]
    fh, fw = np.round(h * factor / 2), np.round(h * factor / 2)
    # [n, 1]
    x1, y1, x2, y2 = np.split(boxes, 4, -1) 
    x1 -= fw
    y1 -= fh
    x2 += fw
    y2 += fh
    boxes = np.concatenate([x1, y1, x2, y2], -1).astype(np.int32)
    return boxes
