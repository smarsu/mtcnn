# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

"""Build mtcnn network."""
from tqdm import tqdm
import math
import numpy as np
import smnet as sm

from model import pnet, pnet_loss
from overlap import bbox_overlap


class PNet(object):
    def __init__(self, 
                 batch_size=32,
                 iou_thrs=0.5,
                 scale_factor=0.79,
                 max_size=512):
        """Init PNet   

        As we use param 'SAME' in the max pool layer, the receptive field
        is 13 rather than 12. 
        
        See https://blog.csdn.net/weixin_42856086/article/details/96456386 
        for details.
        """
        self.batch_size = batch_size
        self.iou_thrs = iou_thrs
        self.scale_factor = scale_factor
        self.max_size = max_size

        self.prior_boxes = {}
        self.cell_size = 13
        self.sizelist = self._get_size_list(self.max_size)

        self._setup()

    
    def _setup(self):
        """Prebuild the pnet network."""
        x = sm.Tensor()
        gt_conf = sm.Tensor()
        gt_box = sm.Tensor() 
        gt_landmark = sm.Tensor() 
        conf_mask = sm.Tensor() 
        box_mask = sm.Tensor() 
        landmark_mask = sm.Tensor()
        conf, box, landmark = pnet(x)
        conf_loss, box_loss, landmark_loss = pnet_loss(conf, box, landmark, 
                                                       gt_conf, gt_box, 
                                                       gt_landmark, conf_mask, 
                                                       box_mask, landmark_mask)


    def _get_size_list(self, max_size):
        """Get size list with the limit of max size."""
        sizelist = []
        size = 12
        while size < max_size:
            sizelist.append((size, size))
            size *= self.scale_factor

        return sizelist


    def resize(self, image, size):
        """Resize image size to `size`
        
        Keep the aspect ratio with avoiding longer size not overflow.
        
        Args:
            image: numpy array.
            size: (int, int)
        """
        h, w = image.shape[:2]
        scale = min(size[0] / h, size[1] / w)
        scale_h, scale_w = h * scale, w *scale
        image = cv2.resize(image, (scale_w, scale_h))
        return image, scale


    def _preprocess(self, image, size):
        """Preprocess input data.
        
        Args:
            image: str or numpy array. If str, read the image to numpy.
            size: (int, int), h, w
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        # 1. Resize and pad the image to `size`
        image, scale = self.resize(image, size)

        # 2. normalize
        image = image / 127.5 - 1.

        # 3. pad
        h, w = image.shape[:2]
        zeros = np.zeros(shape=(size[0], size[1], 3), dtype=np.float32)
        zeros[:h, :w] = image
        image = zeros

        return image, scale


    def _get_prior_box(self, size):
        """Create pnet prior box.

        Attention:
            For size to be odd, it will pad of the left/top in the maxpool 
            stage. So the prior box will switch left for 1 pixel.
            As we use param 'SAME' in the max pool layer, the receptive field
            is 13 rather than 12. 
            
            See https://blog.csdn.net/weixin_42856086/article/details/96456386 
            for details.
        
        Args:
            size: (int, int), the input image size for network.
        """
        size = tuple(size)
        if size not in self.prior_boxes:
            h, w = [math.ceil((dim - 2) / 2) - 4 for dim in size]
            x1 = 2 * np.arange(w) - 1 if w % 2 == 1 else 0
            x2 = x1 + self.cell_size - 1
            y1 = 2 * np.arange(h) - 1 if h % 2 == 1 else 0
            y2 = y1 + self.cell_size - 1

            x1 = np.tile(x1.reshape(1, w), (h, 1))
            x2 = np.tile(x2.reshape(1, w), (h, 1))
            y1 = np.tile(y1.reshape(h, 1), (1, w))
            y2 = np.tile(y2.reshape(h, 1), (1, w))
            prior_boxes = np.stack([x1, y1, x2, y2], -1).reshape(-1, 4)

            self.prior_boxes[size] = prior_boxes
        
        return self.prior_boxes[size]


    def _process_labels(self, labels, scales, size):
        """Process labels.
        
        Parse box label to network input with different scales.

        Args:
            labels: numpy array, batch_size * list of box
            scales: numpy array, batch_size * 1
            size: (int, int), h, w

        Returns:
            confidences: numpy array, [batch, h, w, 2]
            bbox_offsets: numpy array, [batch, h, w, 4]
        """
        # To avoid modify the src box info.
        labels = labels.copy()

        # 1. Scaled label
        labels = labels * scales

        # [h * w, 4]
        h, w = size
        prior_boxes = self._get_prior_box(size)
        confidences = []
        bbox_offsets = []
        for boxes in labels:
            # [h * w, n]
            overlap_scores = bbox_overlap(prior_boxes, boxes)
            overlap_ids = np.argmax(overlap_scores, -1)  # [h * w]
            overlap_scores = np.max(overlap_scores, -1)  # [h * w]
            keep = overlap_scores > self.iou_thrs  # [h * w]

            confidence = keep.reshape(h, w)
            confidence = np.stack([conf_mask, 1 - conf_mask], -1)

            correspond_bbox = boxes[overlap_ids]
            bbox_offset = (correspond_bbox - prior_boxes) / self.cell_size
            bbox_offset = bbox_offset.reshape(h, w, 4)

            # TODO: landmark label
            confidences.append(confidence)
            bbox_offsets.append(bbox_offset)
        
        return np.array(confidences), np.array(bbox_offsets)


    def _check_label_value(self, confidence):
        """If no face valid in labels, pass it.
        
        Args:
            confidence: [n, h, w, 2]

        Return:
            bool, if true, face in labels, good label, else no face in label.
        """
        confidence = confidence[..., 0]
        face_cnt = np.sum(confidence)
        return face_cnt > 0


    def _create_mask(self, confidences):
        """Create conf, box, and landmark mask.
        
        Args:
            confidences: [n, h, w, 2]
        """
        n, h, w, _ = confidences.shape

        confidence = confidence[..., 0]
        face_cnt = np.sum(confidence)
        no_face_cnt = confidence.size - face_cnt

        conf_mask = np.where(confidence==1, 1/face_cnt, 1/no_face_cnt)
        box_mask = 0.25 * 0.5 * 1 / face_cnt
        box_mask = np.full(box_mask, [n, h, w, 4])

        return conf_mask, box_mask


    def train(self, train_datas, epoch, lr):
        """Train pnet.
        
        Args:
            train_datas: list or generator.
            epoch: int
            lr: float
        """
        for step in range(epoch):
            pbar = tqdm(train_datas)
            for datas, labels in train_datas:
                for size in self.sizelist:
                    images, scales = list(zip(*[self._preprocess(data, size) 
                                                for data in datas]))
                    confidences, bbox_offsets = self._process_labels(labels, 
                                                                     scales, 
                                                                     size)
                    if not self._check_label_value(confidences):
                        pass
                    conf_mask, box_mask = self._create_mask(confidences)
