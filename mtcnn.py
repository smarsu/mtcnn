# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

"""Build mtcnn network."""
import time
from tqdm import tqdm
import math
import cv2
import numpy as np
import smnet as sm

from model import pnet, pnet_loss
from overlap import bbox_overlap


class PNet(object):
    def __init__(self, 
                 batch_size=32,
                 iou_thrs=0.5,
                 scale_factor=0.79,
                 max_size=224,):
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
        #print(self.sizelist)

        self._setup()
        self.sess = sm.Session()

    
    def _setup(self):
        """Prebuild the pnet network."""
        self.x = sm.Tensor()
        self.gt_conf = sm.Tensor()
        self.gt_box = sm.Tensor() 
        self.gt_landmark = sm.Tensor() 
        self.conf_mask = sm.Tensor() 
        self.box_mask = sm.Tensor() 
        self.landmark_mask = sm.Tensor()
        self.conf, self.box, self.landmark = pnet(self.x)
        self.conf_loss, self.box_loss, self.landmark_loss = pnet_loss(
            self.conf, self.box, self.landmark, 
            self.gt_conf, self.gt_box, 
            self.gt_landmark, self.conf_mask, 
            self.box_mask, self.landmark_mask)


    def _get_size_list(self, max_size):
        """Get size list with the limit of max size."""
        sizelist = []
        size = 12
        while size < max_size:
            sizelist.append((round(size), round(size)))
            size /= self.scale_factor

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
        # use round to keep accurate
        scale_h, scale_w = round(h * scale), round(w *scale)
        image = cv2.resize(image, (scale_w, scale_h))
        return image, scale


    def _preprocess(self, image, size):
        """Preprocess input data.
        
        Args:
            image: str or numpy array. If str, read the image to numpy.
            size: (int, int), h, w
        """
        #print(image)
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

        Returns:
            prior_boxes: numpy array, [h, w, 4]
        """
        size = tuple(size)
        if size not in self.prior_boxes:
            h, w = [math.ceil((dim - 2) / 2) - 4 for dim in size]
            x1 = 2 * np.arange(w) - (1 if w % 2 == 1 else 0)
            x2 = x1 + self.cell_size - 1
            y1 = 2 * np.arange(h) - (1 if h % 2 == 1 else 0)
            y2 = y1 + self.cell_size - 1

            x1 = np.tile(x1.reshape(1, w), (h, 1))
            x2 = np.tile(x2.reshape(1, w), (h, 1))
            y1 = np.tile(y1.reshape(h, 1), (1, w))
            y2 = np.tile(y2.reshape(h, 1), (1, w))
            prior_boxes = np.stack([x1, y1, x2, y2], -1).reshape(h, w, 4)

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
        prior_boxes = self._get_prior_box(size)
        h, w, _ = prior_boxes.shape
        confidences = []
        bbox_offsets = []
        landmarks = []
        for boxes in labels:
            # [h * w, n]
            #print('prior shape:',prior_boxes.shape)
            #print(boxes.shape)
            overlap_scores = bbox_overlap(prior_boxes, boxes)
            #print(overlap_scores.shape)
            overlap_ids = np.argmax(overlap_scores, -1)  # [h * w]
            overlap_scores = np.max(overlap_scores, -1)  # [h * w]
            keep = overlap_scores > self.iou_thrs  # [h * w]

            confidence = keep.reshape(h, w)
            confidence = np.stack([confidence, 1 - confidence], -1)

            correspond_bbox = boxes[overlap_ids]
            bbox_offset = (correspond_bbox - prior_boxes) / self.cell_size
            bbox_offset = bbox_offset.reshape(h, w, 4)

            # TODO: landmark label
            landmark = np.empty(shape=(h, w, 10))

            confidences.append(confidence)
            bbox_offsets.append(bbox_offset)
            landmarks.append(landmark)
        
        return np.array(confidences), np.array(bbox_offsets), np.array(landmarks)


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

        confidences = confidences[..., 0]
        face_cnt = np.sum(confidences)
        no_face_cnt = confidences.size - face_cnt

        conf_mask = np.where(confidences==1, 0.5/face_cnt, 0.5/no_face_cnt)
        conf_mask = np.stack([conf_mask] * 2, -1)
        scale = 0.25 * 0.5 * 1 / face_cnt
        box_mask = np.stack([scale * confidences] * 4, -1)
        landmark_mask = np.stack([scale * confidences] * 10, -1)

        return conf_mask, box_mask, landmark_mask


    def train(self, train_datas, epoch, lr):
        """Train pnet.
        
        Args:
            train_datas: list or generator.
            epoch: int
            lr: float
        """
        for step in range(epoch):
            #print('Step:', step)
            conf_losses, box_losses, landmark_losses = [], [], []
            for size in self.sizelist[::-1]:
                #print('size:', size)
                pbar = tqdm(train_datas(self.batch_size))
                for datas, labels in pbar:
                    t1 = time.time()
                    images, scales = list(zip(*[self._preprocess(data, size) 
                                                for data in datas]))
                    t2 = time.time()
                    images, scales = np.array(images), np.array(scales)
                    t3 = time.time()
                    #print(images)
                    #print(images.shape)
                    #print(scales)
                    #print(scales.shape)
                    confidences, bbox_offsets, landmarks = self._process_labels(
                                                                    labels, 
                                                                    scales, 
                                                                    size)
                    t4 = time.time()
                    #print(confidences)
                    #print(bbox_offsets)
                    #pbar.set_description('time: {}'.format((t2 - t1, t3 - t2, t4 - t3)))
                    if not self._check_label_value(confidences):
                        continue
                    t5 = time.time()
                    conf_mask, box_mask, landmark_mask = self._create_mask(confidences)
                    t6 = time.time()
                    #print(conf_mask)
                    #print(box_mask)
                    conf_loss, box_loss, landmark_loss = self.sess.forward([
                                  self.conf_loss, 
                                  self.box_loss, 
                                  self.landmark_loss], 
                                 {self.x: images, 
                                  self.gt_conf: confidences,
                                  self.gt_box: bbox_offsets,
                                  self.gt_landmark: landmarks,
                                  self.conf_mask: conf_mask,
                                  self.box_mask: box_mask,
                                  self.landmark_mask: landmark_mask})
                    self.sess.optimize([self.conf_loss, self.box_loss, self.landmark_loss],
                                       lr=lr)
                    conf_loss, box_loss, landmark_loss = np.sum(conf_loss), np.sum(box_loss), np.sum(landmark_loss)
                    conf_losses.append(conf_loss) 
                    box_losses.append(box_loss) 
                    landmark_losses.append(landmark_loss)

                    pbar.set_description('step: {}, size: {}, conf loss: {}, '
                                        'box loss: {}, landmark loss: {}'.format(
                                            step, size, np.mean(conf_losses), np.mean(box_losses), np.mean(landmark_losses)))
                    

