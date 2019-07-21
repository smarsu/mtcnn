# --------------------------------------------------------
# Square
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np


def square_boxes(boxes):
    """Make box to square.
    
    Args:
        boxes: ndarray, [n, 4], [x1, y1, x2, y2]
    """
    boxes = np.round(boxes.copy())

    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    h = y2 - y1 + 1  # [n]
    w = x2 - x1 + 1  # [n]

    max_size = np.maximum(h, w)  # [n]
    
    offset_top = (max_size - h) // 2  # [n]
    offset_bottom = (max_size - h) - offset_top
    offset_left = (max_size - w) // 2
    offset_right = (max_size - w) - offset_left

    x1 -= offset_left
    y1 -= offset_top
    x2 += offset_right
    y2 += offset_bottom

    return np.stack([x1, y1, x2, y2], -1)


if __name__ == '__main__':
    boxes = [[1, 2, 3, 4], [0, 0, 1, 1], [6, 6, 8, 9], [-1, -3, 6, -2]]
    print(square_boxes(boxes))
