# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np
from datasets import WiderFace
from mtcnn import PNet

np.random.seed(196)


if __name__ == '__main__':
    parse = 'train'
    net = 'pnet'

    if parse == 'train':
        widerface = WiderFace('/datasets/wider/images', 
                              '/datasets/wider/wider_face_split')

        if net == 'pnet':
            pnet = PNet()
            #pnet.train(widerface.train_datas, 10, lr=1)
            pnet.train(widerface.train_datas_debug, 10, lr=1)
            
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
