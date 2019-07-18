# --------------------------------------------------------
# Face Datasets
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
import numpy as np


class Dataset(object):
    """The base class of dataset."""
    def __init__(self):
        pass


    @property
    def size(self):
        """Return the number of train datas."""
        raise NotImplementedError


    def train_datas(self, batch_size):
        """Yield batch size train datas per step. 

        Train datas should be shuffled.
        
        Args:
            batch_size: int, > 0
        """
        if not isinstance(batch_size, int):
            raise ValueError('In Dataset, batch_size should be int, get '
                             '{}'.format(type(batch_size)))
        if batch_size <= 0:
            raise ValueError('In Dataset, batch_size should larger equal to '
                             '1, get {}'.format(batch_size))
        
        indices = list(range(self.size))
        np.random.shuffle(indices)

        epoch_size = self.size // batch_size * batch_size
        self._train_datas = self._train_datas[indices][:epoch_size]  # [epoch_size, ...]
        self._train_labels = self._train_labels[indices][:epoch_size] # [epoch_size, ...]
        
        datas = []
        for i in range(self.size // batch_size):
            # for label, we have box and landmark which is 0.
            datas.append([self._train_datas[i*batch_size:(i+1)*batch_size], 
                          self._train_labels[i*batch_size:(i+1)*batch_size]])
        return datas


class WiderFace(Dataset):
    def __init__(self, train_image_path, label_path, value_image_path=None, 
                 test_image_path=None):
        """
        TODO(smarsu): Add way to read `value_image_path` and `test_image_path`.
                      Add way to read `value_label_path` and `test_label_path`.

        Args:
            train_image_path: str, the path of train images.
            label_path: str
        """
        self.train_image_path = train_image_path
        self.label_path = label_path
        self.train_label_path = osp.join(self.label_path, 
                                          'wider_face_train_bbx_gt.txt')

        self._train_datas, self._train_labels = self._read_train_datas()


    @property
    def size(self):
        """Return the number of train datas.
        
        Assert the size of self._train_datas and self._train_labels is equal.
        """
        return len(self._train_datas)


    def _real_image_path(self, path):
        """Get real path of image.

        self.train_image_path + '/' + path
        
        Args:
            path: str, the image name(id) of labels.
        """
        return osp.join(self.train_image_path, path)


    def _read_train_datas(self):
        """The special way to read wider face labels.

        Args:
            label_path: str, 
        """
        with open(self.train_label_path, 'r') as fb:
            lines = fb.readlines()
            return self._parse_raw_labels(lines)


    def _parse_raw_labels(self, lines):
        """Parse raw str lines to python object.
        
        Args:
            lines: list of str, with the structure of 
                File name
                Number of bounding box
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

        Returns:
            images: numpy array, [n], image paths
            labels: numpy array, [n, 4], [x1, y1, x2, y2]
        """
        images = []
        labels = []
        idx = 0
        while idx < len(lines):
            image_path = lines[idx]
            images.append(image_path)
            idx += 1

            num = int(lines[idx])
            idx += 1

            labels_ = []
            for _ in range(num):
                x1, y1, w, h, blur, expression, illumination, invalid, \
                    occlusion, pose = [int(v) 
                                       for v in lines[idx].strip().split()]
                x2, y2 = x1 + w - 1, y1 + h - 1  # -1 to get the read x2, y2

                labels_.append([x1, y1, x2, y2])
                idx += 1
        
            labels.append(np.array(labels_))
        return np.array(images), np.array(labels)


if __name__ == '__main__':
    import time
    # Test wider face dataset
    wider = WiderFace('/datasets/wider/images', 
                      '/datasets/wider/wider_face_split')
    t1 = time.time()
    for data, label in wider.train_datas(1):
        print(data, label)
    t2 = time.time()
    print('Time for read wider dataset:', t2 - t1)  # 3.6073758602142334s with `print`