# --------------------------------------------------------
# IOU
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np


def bbox_overlap(bbox1, bbox2):
    """Compute Intersection over Union of bbox1 and bbox2.
    
    bbox coord layout [x1, y1, x2, y2]

    Args:
        bbox1: numpy array, [n, 4]
        bbox2: numpy array, [m, 4]

    Returns:
        iou: [n, m]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = np.split(bbox1, 4, -1)  # [n, 1]
    b2_x1, b2_y1, b2_x2, b2_y2 = np.split(bbox2, 4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x1.T, b2_y1.T, b2_x2.T, b2_y2.T  # [1, m]

    area1 = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    area2 = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # [n, m]
    inter_area = np.maximum((np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1) + 1), 0) * \
                 np.maximum((np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1) + 1), 0)

    return inter_area / (area1 + area2 - inter_area)


if __name__ == '__main__':
    bbox1 = np.array([[0, 0, 1, 1]]) 
    bbox2 = np.array([[1, 1, 2, 2]])
    print(bbox_overlap(bbox1, bbox2))

    bbox1 = np.array([[0, 0, 1, 1]]) 
    bbox2 = np.array([[2, 2, 3, 3]])
    print(bbox_overlap(bbox1, bbox2))

    bbox1 = np.array([[0, 0, 1, 1]]) 
    bbox2 = np.array([[0, 0, 1, 1]])
    print(bbox_overlap(bbox1, bbox2))

    bbox1 = np.array([[0, 0, 1, 1], [1, 2, 7, 6]]) 
    bbox2 = np.array([[0, 0, 1, 1], [2, 4, 5, 6], [1, 2, 7, 6]])
    print(bbox_overlap(bbox1, bbox2))
