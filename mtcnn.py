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

from model import pnet, pnet_loss, rnet, rnet_loss, onet, onet_loss
from overlap import bbox_overlap
from nms import nms
from square import square_boxes


class PNet(object):
    def __init__(self, 
                 batch_size=1,
                 iou_thrs=0.5,
                 nms_thrs=0.5,
                 conf_thrs=0.5,
                 min_face=20,
                 scale_factor=0.79,
                 min_size=12,
                 max_size=224,
                 no_mask=False,
                 alpha=(1., 0.5, 0.5),
                 rd_size=False,
                 src_size=False,
                 nms_topk=None,
                 model_root='data/model',
                 demo_root='data/demo'):
        """Init PNet   

        As we use param 'SAME' in the max pool layer, the receptive field
        is 13 rather than 12. 
        
        See https://blog.csdn.net/weixin_42856086/article/details/96456386 
        for details.
        """
        self.batch_size = batch_size
        self.iou_thrs = iou_thrs
        self.nms_thrs = nms_thrs
        self.conf_thrs = conf_thrs
        self.min_face = min_face
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.max_size = max_size
        self.no_mask = no_mask
        self.rd_size = rd_size
        self.src_size = src_size
        self.alpha = alpha
        self.nms_topk = nms_topk
        self.model_root = model_root
        self.demo_root = demo_root

        self.neg_thrs = 0.3
        self.pos_thrs = 0.65
        self.part_thrs = 0.4

        self.prior_boxes = {}
        self.cell_size = 13
        self.sizelist = self._get_size_list(self.min_size, self.max_size)
        #print(self.sizelist)

        self._setup()
        self.sess = sm.Session()
        print(self.sess._variables)

    
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


    def _get_size_list(self, min_size, max_size):
        """Get size list with the limit of max size."""
        sizelist = []
        size = min_size
        while size < max_size:
            sizelist.append((round(size), round(size)))
            size /= self.scale_factor

        return sizelist


    def get_max_size(self, images):
        """
        Args:
            images: list of ndarray
        """
        sizes = [image.shape[:2] for image in images]  # [n, 2]
        h, w = np.max(sizes, 0)  # [2]
        return h, w


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
        scale_h, scale_w = int(round(h * scale)), int(round(w * scale))
        image = cv2.resize(image, (scale_w, scale_h))
        return image, scale


    def _preprocess(self, image, size):
        """Preprocess input data.
        
        Args:
            image: str or numpy array. If str, read the image to numpy.
            size: (int, int), h, w
        """
        size = (int(round(size[0])), int(round(size[1])))
        #print(image)
        #if isinstance(image, str):
        image = cv2.imread(image) if isinstance(image, str) else image
        #else:
            #raise ValueError('image must be a str')

        # 1. Resize and pad the image to `size`
        image, scale = self.resize(image, size)

        # 2. normalize
        image = (image / 127.5) - 1

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
            #debug
            print('Create prior box with size {}'.format(size))
        
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
        # 1. Scaled label
        # To avoid modify the src box info.
        size = [int(round(size[0])), int(round(size[1]))]
        scaled_labels = labels * scales
        labels = scaled_labels

        # [h * w, 4]
        prior_boxes = self._get_prior_box(size)
        h, w, _ = prior_boxes.shape
        confidences = []
        bbox_offsets = []
        landmarks = []
        conf_masks = []
        box_masks = []
        landmark_masks = []
        for boxes in labels:
            # [h * w, n]
            #print('prior shape:',prior_boxes.shape)
            #print(boxes.shape)
            overlap_scores = bbox_overlap(prior_boxes, boxes)
            #print(overlap_scores.shape)
            overlap_ids = np.argmax(overlap_scores, -1)  # [h * w]
            overlap_scores = np.max(overlap_scores, -1)  # [h * w]
            keep = overlap_scores > self.pos_thrs  # [h * w]

            confidence = keep.reshape(h, w)
            confidence = np.stack([confidence, 1 - confidence], -1)

            correspond_bbox = boxes[overlap_ids]
            bbox_offset = (correspond_bbox - prior_boxes) / self.cell_size
            bbox_offset = bbox_offset.reshape(h, w, 4)

            # TODO: landmark label
            landmark = np.zeros(shape=(h, w, 10))

            # mask
            pos_mask = (overlap_scores > self.pos_thrs).astype(np.float32)
            neg_mask = (overlap_scores < self.neg_thrs).astype(np.float32)
            part_mask = (overlap_scores > self.part_thrs).astype(np.float32)

            pos_cnt = np.sum(pos_mask)
            neg_cnt = np.sum(neg_mask)
            part_cnt = np.sum(part_mask)
            if pos_cnt > 0:
                pos_mask /= pos_cnt
            if neg_cnt > 0:
                neg_mask /= neg_cnt
            if part_cnt > 0:
                part_mask /= part_cnt

            conf_mask = self.alpha[0] * (0.5 * pos_mask + 0.5 * neg_mask)
            box_mask = self.alpha[1] * 0.25 * part_mask
            conf_mask = np.stack([conf_mask] * 2, -1)
            box_mask = np.stack([box_mask] * 4, -1)
            landmark_mask = self.alpha[2] * np.zeros([h, w, 10])
            # debug
            #box_mask = self.alpha[1] * np.zeros([h, w, 4])

            confidences.append(confidence)
            bbox_offsets.append(bbox_offset)
            landmarks.append(landmark)
            conf_masks.append(conf_mask)
            box_masks.append(box_mask)
            landmark_masks.append(landmark_mask)
        
        return np.array(confidences), np.array(bbox_offsets), np.array(landmarks), \
               np.array(conf_masks), np.array(box_masks), np.array(landmark_masks)


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
        if self.no_mask:
            n, h, w, _ = confidences.shape
            conf_mask = self.alpha[0] * np.ones_like(confidences) / self.batch_size
            box_mask =  self.alpha[1] * np.stack([confidences[..., 0]] * 4, -1) / self.batch_size  # Here * 0.25 before.
            landmark_mask = self.alpha[2] * np.zeros([n, h, w, 10]) / self.batch_size

        else:
            n, h, w, _ = confidences.shape

            confidences = confidences[..., 0]
            face_cnt = np.sum(confidences)
            no_face_cnt = confidences.size - face_cnt
    
            conf_mask = np.where(confidences==1, 1/face_cnt, 1/no_face_cnt)
            conf_mask = np.stack([conf_mask] * 2, -1)
            box_mask = np.stack([confidences / face_cnt] * 4, -1)
            landmark_mask = np.zeros([n, h, w, 10])  # np.stack([scale * confidences] * 10, -1)

        return conf_mask, box_mask, landmark_mask


    def train(self, train_datas, epoch, lr, weight_decay=5e-4):
        """Train pnet.
        
        Args:
            train_datas: list or generator.
            epoch: int
            lr: float
        """
        for step in range(epoch):
            #print('Step:', step)
            conf_losses, box_losses, landmark_losses = [], [], []
            pbar = tqdm(train_datas(self.batch_size))
            for datas, labels in pbar:
                datas = [cv2.imread(image) for image in datas]
                size = list(self.get_max_size(datas))
                scale = 12 / self.min_face
                size[0] *= scale
                size[1] *= scale
                #while min([int(round(size[0])), int(round(size[1]))]) >= 12:
                for size in self.sizelist:
                #print('size:', size)
                    #if self.rd_size:
                    #    size = self.sizelist[np.random.choice(len(self.sizelist))]
                    #if self.src_size:
                    #    datas = [cv2.imread(image) for image in datas]
                    #    size = self.get_max_size(datas)
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
                    confidences, bbox_offsets, landmarks, conf_mask, box_mask, landmark_mask = self._process_labels(
                                                                    labels, 
                                                                    scales, 
                                                                    size)
                    t4 = time.time()
                    #print(confidences)
                    #print(bbox_offsets)
                    # debuf
                    #size[0] *= self.scale_factor
                    #size[1] *= self.scale_factor
                    if not self._check_label_value(confidences):
                        continue
                    t5 = time.time()
                    #conf_mask, box_mask, landmark_mask = self._create_mask(confidences)
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
                                       lr=lr, weight_decay=weight_decay)
                    t7 = time.time()
                    conf_loss, box_loss, landmark_loss = np.sum(conf_loss), np.sum(box_loss), np.sum(landmark_loss)
                    conf_losses.append(conf_loss) 
                    box_losses.append(box_loss) 
                    landmark_losses.append(landmark_loss)
                    t8 = time.time()

                    # debug
                    #print('time: {}'.format((t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7)))
                    times = (t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7)
                    times = [round(time * 1000) for time in times]

                    pbar.set_description('step: {}, conf loss: {}, '
                                        'box loss: {}, landmark loss: {}, time: {}'.format(
                                            step, np.mean(conf_losses), np.mean(box_losses), np.mean(landmark_losses),
                                            times))

            if self.rd_size:
                size = 'rand_size'
            size = 'cycle'
            total_loss = np.mean(conf_losses) + np.mean(box_losses) + np.mean(landmark_losses)
            self.sess.save('/'.join([self.model_root, '{}_{}_{}_{}_{}'.format(str(total_loss), size, step, lr, 'pnet')]))

            if self.rd_size:
                break


    def test(self, image):
        """Test pnet.
        
        Args:
            image: str, path.
        """
        #debuf
        #print(image)
        image = cv2.imread(image)
        confs, boxs = [], []
        size = list(image.shape[:2])
        scale = 12 / self.min_face
        size[0] *= scale
        size[1] *= scale
        while min(size) >= 12:
            images, scales = list(zip(*[self._preprocess(image, size)]))
            # [1, h, w, 3]
            images, scales = np.array(images), np.array(scales)

            conf, box, landmark = self.sess.forward([
                            self.conf, 
                            self.box, 
                            self.landmark], 
                            {self.x: images, 
                            self.gt_conf: 0,
                            self.gt_box: 0,
                            self.gt_landmark: 0,
                            self.conf_mask: 0,
                            self.box_mask: 0,
                            self.landmark_mask: 0})
            conf = conf.reshape(-1, 2)
            # softmax
            conf = np.exp(conf - np.max(conf, -1, keepdims=True))
            conf = conf / np.sum(conf, -1, keepdims=True)
            box = box.reshape(-1, 4)
            box = self._regbox(box, size, scales)
            conf, box = self._postprocess(conf, box)  # NMS before the last nms
            confs.append(conf)
            boxs.append(box)
            #print(conf.shape)

            size[0] *= self.scale_factor
            size[1] *= self.scale_factor

        confs = np.concatenate(confs, 0)
        boxs = np.concatenate(boxs, 0)
        confs, boxs = self._postprocess(confs, boxs)
        return confs, boxs

    
    def _regbox(self, box, size, scale):
        """Regress box."""
        size = (round(size[0]), round(size[1]))
        box = box.reshape(-1, 4)
        priorbox = self._get_prior_box(size).reshape(-1, 4)
        regbox = (box * self.cell_size + priorbox) / scale
        return regbox

        
    def _postprocess(self, confidence, box):
        """Postprocess the result of network to bboxes.
        
        Args:
            confidence: numpy array, [h, w, 2]
            box: numpy array, [h, w, 4]

        Return:
            conf: [-1, 2]
            box: [-1, 4]
        """
        confidence = confidence.reshape(-1, 2)
        box = box.reshape(-1, 4)
        # softmax
        #confidence = np.exp(confidence - np.max(confidence, -1, keepdims=True))
        #confidence = confidence / np.sum(confidence, -1, keepdims=True)
        #keep = confidence[..., 0] >= confidence[..., 1]
        keep = confidence[..., 0] > self.conf_thrs
        
        #print(confidence[..., 0])
        #print(confidence.shape)

        conf = confidence[..., 0][keep]
        box = box[keep]

        keep = nms(np.concatenate([box, conf.reshape(-1, 1)], -1), self.nms_thrs, self.nms_topk)

        conf = np.stack([conf[keep]] * 2, -1)
        box = box[keep]
        #print(conf.shape)
        #print(box.shape)

        return conf, box


    def _check_model_capable(self, train_datas):
        datas, labels = train_datas(1)[0]
        confs, boxs = [], []
        for size in self.sizelist[::-1]:
            images, scales = list(zip(*[self._preprocess(data, size) 
                                        for data in datas]))
            images, scales = np.array(images), np.array(scales)
            conf, box, landmarks = self._process_labels(labels, 
                                                        scales, 
                                                        size)
            #if not self._check_label_value(confidences):
            #    raise ValueError('Invalid image')
            
            conf = conf.reshape(-1, 2)
            box = box.reshape(-1, 4)
            box = self._regbox(box, size, scales)
            confs.append(conf)
            boxs.append(box)
            print(conf.shape)

        confs = np.concatenate(confs, 0)
        boxs = np.concatenate(boxs, 0)
        confs, boxs = self._postprocess(confs, boxs)
        print('true label:', labels)
        return confs, boxs


class RNet(object):
    def __init__(self, 
                 batch_size=32,
                 input_size=(24, 24),
                 iou_thrs=0.65,
                 conf_thrs=0.5,
                 nms_thrs=0.5,
                 scale_mask=True,
                 model_root='data/model'):
        """Init RNet"""
        self.batch_size = batch_size
        self.input_size = input_size
        self.iou_thrs = iou_thrs
        self.conf_thrs = conf_thrs
        self.nms_thrs = nms_thrs
        self.scale_mask = scale_mask
        self.model_root = model_root

        self.neg_thrs = 0.3
        self.pos_thrs = 0.65
        self.part_thrs = 0.4

        self.cell_size = input_size[0]

        sm.reset_default_graph()
        self.sess = sm.Session()
        self._setup()
        #self.sess = sm.Session()

    
    def _setup(self):
        """Prebuild the rnet network."""
        self.x = sm.Tensor()
        self.gt_conf = sm.Tensor()
        self.gt_box = sm.Tensor() 
        self.gt_landmark = sm.Tensor() 
        self.conf_mask = sm.Tensor() 
        self.box_mask = sm.Tensor() 
        self.landmark_mask = sm.Tensor()
        self.conf, self.box, self.landmark = rnet(self.x)
        self.conf_loss, self.box_loss, self.landmark_loss = rnet_loss(
            self.conf, self.box, self.landmark, 
            self.gt_conf, self.gt_box, 
            self.gt_landmark, self.conf_mask, 
            self.box_mask, self.landmark_mask)


    def _preprocess(self, datas, pnet_boxes):
        """
        Attention:
            We assum x1 < right border, y1 < bottom boder, x2 > 0, y2 > 0

        Args:
            datas: list of str
            pnet_boxes: ndarray, [n, 4] assert the pnet box have been pad to square.

        Returns:
            images: ndarray, [n, 24, 24, 3]
        """
        images = []
        for data, box in zip(datas, pnet_boxes):
            #box = box[0]
            data = cv2.imread(data) if isinstance(data, str) else data
            img_h, img_w, _ = data.shape
            
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            assert h == w

            zeros = np.zeros(shape=(h, w, 3))
            left = -x1 if x1 < 0 else 0
            top = -y1 if y1 < 0 else 0
            right = -(x2 - img_w) if x2 > img_w else w
            bottom = -(y2 - img_h) if y2 > img_h else h

            x1 = min(max(x1, 0), img_w)
            y1 = min(max(y1, 0), img_h)
            x2 = max(min(x2, img_w), 0)
            y2 = max(min(y2, img_h), 0)

            zeros[top:bottom, left:right] = (data[y1:y2, x1:x2] / 127.5 - 1.) 
            #zeros[top:bottom, left:right] = data[y1:y2, x1:x2]
            
            image = cv2.resize(zeros, self.input_size)
            #image = image / 127.5 - 1.
            #cv2.imwrite(str(box) + '.jpg', cv2.resize(image.astype(np.uint8), (224, 224)))
            images.append(image)

        return np.array(images)


    def _parse_labels(self, datas, pnet_boxes, gtboxes):
        """Parse boxes of pnet output and ground truth box.
        
        Args:
            datas: strs or ndarray, [n, ...]
            pnet_boxes: ndarray, [n, 4]
            gtboxes:
        """
        confidences = []
        bbox_offsets = []
        landmarks = []
        conf_masks = []
        box_masks = []
        landmark_masks = []
        for data, pnet_boxe in zip(datas, pnet_boxes):
            #pnet_boxe = pnet_boxe[0]
            px1, py1, px2, py2 = pnet_boxe
            ph = py2 - py1
            pw = px2 - px1
            assert ph == pw, 'Assert the pnet box have been pad to square.'

            gtbox = gtboxes(data)  # [m, 4]
            overlap_scores = bbox_overlap(pnet_boxe, gtbox)  # [1, m]
            overlap_ids = np.argmax(overlap_scores, -1)  # [1]
            overlap_scores = np.max(overlap_scores, -1)  # [1]
            keep = overlap_scores > self.iou_thrs

            confidence = keep
            confidence = [confidence, 1 - confidence]  # [2]

            correspond_bbox = gtbox[overlap_ids]
            bbox_offset = (correspond_bbox - pnet_boxe) / [pw, ph, pw, ph]  # [4]

            # TODO: landmark label
            landmark = [0] * 10

            pos_mask = (overlap_scores > self.pos_thrs).astype(np.float32) * 0.25
            neg_mask = (overlap_scores < self.neg_thrs).astype(np.float32) * 0.25
            part_mask = (overlap_scores > self.part_thrs).astype(np.float32) * 0.5
            landmark_mask = [0] * 10

            confidences.append(confidence)
            bbox_offsets.append(bbox_offset)
            landmarks.append(landmark)
            conf_masks.append([pos_mask + neg_mask, pos_mask + neg_mask])
            box_masks.append([part_mask, part_mask, part_mask, part_mask])
            landmark_masks.append(landmark_mask)

        return np.array(confidences).squeeze(), np.array(bbox_offsets).squeeze(), np.array(landmarks).squeeze(), \
               np.array(conf_masks).squeeze(), np.array(box_masks).squeeze(), np.array(landmark_masks).squeeze()


    def _check_label_value(self, confidences):
        """
        Args:
            confidences: ndarray, [n, 2]
        
        Returns:
            bool, if True, valid face label in this step.
        """
        confidences = confidences[..., 0]
        return int(np.sum(confidences))


    def _check_prebox_value(self, pnet_boxes):
        """
        Args:
            pnet_boxes: [n, 4]
        """
        for pbox in pnet_boxes:
            x1, y1, x2, y2 = pbox
            if x1 >= x2 or y1 >= y2:
                return False
        return True


    def _create_mask(self, confidences):
        """
        Args:
            confidences: ndarray, [n, 2]
        """
        if self.scale_mask:
            conf = confidences[..., 0]
            face_cnt = np.sum(conf)
            no_face_cnt = conf.size - face_cnt

            conf_mask = np.where(conf==1, 0.5/face_cnt, 0.5/no_face_cnt)
            conf_mask = np.stack([conf_mask] * 2, -1)
            box_mask = np.stack([0.25 * conf / face_cnt] * 4, -1)
            landmark_mask = np.zeros([self.batch_size, 10])
        else:
            conf = confidences[..., 0]
            conf_mask = 1 * np.ones(shape=[self.batch_size, 2]) / self.batch_size
            box_mask = 0.5 * np.stack([0.5 * conf] * 4, -1)
            landmark_mask = np.zeros([self.batch_size, 10])

        return conf_mask, box_mask, landmark_mask

        
    def train(self, train_datas, epoch, lr, gtboxes, weight_decay=5e-4, stage='rnet'):
        """Train rnet.
        
        Args:
            train_datas: list or generator.
                image path, pnet out, ground true
                [([batch * str], [batch * one label], [batch * list of label]), ...]
            epoch: int
            lr: float
            gtboxes: func, feed image name get list of list of gt_box
        """
        for step in range(epoch):
            conf_losses, box_losses, landmark_losses = [], [], []
            pbar = tqdm(train_datas(self.batch_size))
            for datas, pnet_boxes in pbar:
                #pnet_boxes = pnet_boxes.squeeze()
                pnet_boxes = pnet_boxes[:, 0, :]
                t1 = time.time()
                confidences, bbox_offsets, landmarks, conf_mask, box_mask, landmark_mask = self._parse_labels(
                    datas, pnet_boxes, gtboxes)
                #confidences = confidences.reshape(-1, 1, 1, 2)
                #bbox_offsets = bbox_offsets.reshape(-1, 1, 1, 4)
                #landmarks = landmarks.reshape(-1, 1, 1, 10)
                t2 = time.time()
                if not self._check_label_value(confidences):
                    continue
                if not self._check_prebox_value(pnet_boxes):
                    continue
                t3 = time.time()
                #conf_mask, box_mask, landmark_mask = self._create_mask(confidences)
                #conf_mask = conf_mask.reshape(-1, 1, 1, 2)
                #box_mask = box_mask.reshape(-1, 1, 1, 4)
                #landmark_mask = landmark_mask.reshape(-1, 1, 1, 10)
                t4 = time.time()
                images = self._preprocess(datas, pnet_boxes)
                t5 = time.time()
                #print(images)
                if True:
                    pconf, pbox, plandmark, conf_loss, box_loss, landmark_loss = self.sess.forward([
                                    self.conf,
                                    self.box, 
                                    self.landmark,
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
                                        lr=lr, weight_decay=weight_decay)
                    t6 = time.time()
                    conf_loss, box_loss, landmark_loss = np.sum(conf_loss), np.sum(box_loss), np.sum(landmark_loss)
                    conf_losses.append(conf_loss) 
                    box_losses.append(box_loss) 
                    landmark_losses.append(landmark_loss)
                    t7 = time.time()

                    times = (t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)
                    times = [round(time * 1000) for time in times]
                    #print(images)
                    pbar.set_description('step: {}, conf loss: {}, '
                                        'box loss: {}, land loss: {}, time: {}, gt_conf: {}, conf: {}'.format(
                                            step, np.mean(conf_losses), np.mean(box_losses), np.mean(landmark_losses),
                                            times, confidences.reshape(-1), pconf.reshape(-1)))
            
            total_loss = np.mean(conf_losses) + np.mean(box_losses) + np.mean(landmark_losses)
            self.sess.save('/'.join([self.model_root, '{}_{}_{}_{}'.format(str(total_loss), step, lr, stage)]))


    def test(self, image, pboxes):
        """Test rnet.
        
        Assert the image to be a ndarray to reduce the time of read image.

        Args:
            image: str or ndarray.
            pboxes: box from pnet.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError('image must be np.ndarray')
        if not(len(pboxes.shape) == 2 and pboxes.shape[-1] == 4):
            raise ValueError('pboxes must have shape [n, 4]')

        if not self._check_prebox_value(pboxes):
            return None, [[0, 0, 0, 0]]

        pboxes = square_boxes(pboxes)
        image = cv2.imread(image) if isinstance(image, str) else image
        images = self._preprocess([image] * len(pboxes), pboxes)
        conf, box, landmark = self.sess.forward([
                        self.conf, 
                        self.box, 
                        self.landmark], 
                        {self.x: images, 
                        self.gt_conf: 0,
                        self.gt_box: 0,
                        self.gt_landmark: 0,
                        self.conf_mask: 0,
                        self.box_mask: 0,
                        self.landmark_mask: 0})
        confs, boxes = self._postprocess(conf, box, pboxes)
        return confs, boxes
        

    def _postprocess(self, confs, boxes, pboxes):
        """Post process boxes.
        
        Args:
            confs: [n, 2]
            boxes: [n, 4]
            pboxes: [n, 4]
        """
        # softmax
        confs = np.exp(confs - np.max(confs, -1, keepdims=True))
        confs = confs / np.sum(confs, -1, keepdims=True)     
        #print(boxes)

        keep = confs[..., 0] > self.conf_thrs

        confs = confs[..., 0:1][keep]  # [n, 1]
        boxes = boxes[keep]
        pboxes = pboxes[keep]

        x1, y1, x2, y2 = np.split(pboxes, 4, -1)  # [n, 1], ...
        h, w = y2 - y1, x2 - x1  # [n, 1], ...
        gtboxes = boxes * np.concatenate([w, h, w, h], -1) + pboxes  # [n, 4]

        keep = nms(np.concatenate([gtboxes, confs], -1), self.nms_thrs)

        confs = confs[keep]
        boxes = gtboxes[keep]

        return confs, boxes


class ONet(RNet):
    def __init__(self, 
                 batch_size=32,
                 input_size=(48, 48),
                 iou_thrs=0.65,
                 conf_thrs=0.5,
                 nms_thrs=0.5,
                 scale_mask=True,
                 model_root='data/model'):
        super(ONet, self).__init__(batch_size, input_size, iou_thrs, conf_thrs,
                                   nms_thrs, scale_mask, model_root)


    def _setup(self):
        """Prebuild the onet network."""
        self.x = sm.Tensor()
        self.gt_conf = sm.Tensor()
        self.gt_box = sm.Tensor() 
        self.gt_landmark = sm.Tensor() 
        self.conf_mask = sm.Tensor() 
        self.box_mask = sm.Tensor() 
        self.landmark_mask = sm.Tensor()
        self.conf, self.box, self.landmark = onet(self.x)
        self.conf_loss, self.box_loss, self.landmark_loss = onet_loss(
            self.conf, self.box, self.landmark, 
            self.gt_conf, self.gt_box, 
            self.gt_landmark, self.conf_mask, 
            self.box_mask, self.landmark_mask)
