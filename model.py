# --------------------------------------------------------
# MTCNN
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

"""Implementation of MTCNN in SMNet.

Reference:
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
https://arxiv.org/abs/1604.02878
"""
import numpy as np
import smnet as sm
import tensorflow as tf


def pnet(x):
    """PNet in MTCNN.
    
    Args:
        x: The input tensor. (n, h, w, 3)

    Returns: 
        conf: The output confidence tensor.
        box: The output box tensor.
        landmark: The output landmark tensor.
    """
    # smnet-gpu only support NCHW in conv and pool
    # smnet-cpu only support NHWC in conv and pool
    if sm.net.use_cuda:
        x = sm.transform(x, 'NHWC2NCHW')  
    x = sm.slim.conv2d(x, 3, 10, 3, 1, padding='VALID')
    x = sm.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME')
    x = sm.slim.conv2d(x, 10, 16, 3, 1, padding='VALID')
    x = sm.slim.conv2d(x, 16, 32, 3, 1, padding='VALID')
    x = sm.slim.conv2d(x, 32, 16, 1, 1, padding='VALID', act=None)
    if sm.net.use_cuda:
        x = sm.transform(x, 'NCHW2NHWC')  
    conf, box, landmark = sm.split(x, (2, 6, 16), axis=-1)
    return conf, box, landmark


def pnet_loss(conf, box, landmark, gt_conf, gt_box, gt_landmark, conf_mask, 
              box_mask, landmark_mask):
    """Create the loss of pnet.
    
    Args:
        conf, box, landmark: The output of pnet.
        gt_conf, gt_box, gt_landmark: ground truth labels.
        conf_mask, box_mask, landmark_mask: to balance the three loss
    """
    conf_loss = conf_mask * sm.softmax_cross_entropy_with_logits(labels=gt_conf, 
                                                                 logits=conf)
    box_loss = box_mask * sm.hse(gt_box, box)
    landmark_loss = landmark_mask * sm.hse(gt_landmark, landmark)
    return conf_loss, box_loss, landmark_loss


def rnet(x):
    """RNet in MTCNN.
    
    Args:
        x: The input tensor. (n, 24, 24, 3)
    """
    if sm.net.use_cuda:
        x = sm.transform(x, 'NHWC2NCHW')
    x = sm.slim.conv2d(x, 3, 28, 3, 1, padding='VALID', border=0.01)
    x = sm.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME')
    x = sm.slim.conv2d(x, 28, 48, 3, 1, padding='VALID', border=0.01)
    x = sm.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='VALID')
    x = sm.slim.conv2d(x, 48, 64, 2, 1, padding='VALID', border=0.01)
    # use conv2d to replace full-connect
    x = sm.slim.conv2d(x, 64, 128, 3, 1, padding='VALID', border=0.01)
    x = sm.slim.conv2d(x, 128, 16, 1, 1, padding='VALID', act=None, border=0.01)
    if sm.net.use_cuda:
        x = sm.transform(x, 'NCHW2NHWC')
    #x = sm.reshape(x, (-1, 3 * 3 * 64))
    #x = sm.slim.fc(x, 3 * 3 * 64, 128, bias=True)
    #x = sm.slim.fc(x, 128, 16, bias=True, act=None)
    x = sm.reshape(x, (-1, 16))
    conf, box, landmark = sm.split(x, (2, 6, 16), axis=-1)
    return conf, box, landmark



def rnet_loss(conf, box, landmark, gt_conf, gt_box, gt_landmark, conf_mask, 
              box_mask, landmark_mask):
    """Create the loss of rnet.
    
    Args:
        conf, box, landmark: The output of pnet.
        gt_conf, gt_box, gt_landmark: ground truth labels.
        conf_mask, box_mask, landmark_mask: to balance the three loss
    """
    conf_loss = conf_mask * sm.softmax_cross_entropy_with_logits(labels=gt_conf, 
                                                                 logits=conf)
    #conf_loss = conf_mask * sm.hse(gt_conf, conf)
    box_loss = box_mask * sm.hse(gt_box, box)
    landmark_loss = landmark_mask * sm.hse(gt_landmark, landmark)
    return conf_loss, box_loss, landmark_loss


def onet(x):
    """ONet in MTCNN.
    
    Args:
        x: The input tensor. (n, 48, 48, 3)
    """
    if sm.net.use_cuda:
        x = sm.transform(x, 'NHWC2NCHW')
    x = sm.slim.conv2d(x, 3, 32, 3, 1, padding='VALID')
    x = sm.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME')
    x = sm.slim.conv2d(x, 32, 64, 3, 1, padding='VALID')
    x = sm.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='VALID')
    x = sm.slim.conv2d(x, 64, 64, 3, 1, padding='VALID')
    x = sm.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
    x = sm.slim.conv2d(x, 64, 128, 2, 1, padding='VALID')
    # use conv2d to replace full-connect
    x = sm.slim.conv2d(x, 128, 256, 3, 1, padding='VALID')
    x = sm.slim.conv2d(x, 256, 16, 1, 1, padding='VALID', act=None)
    if sm.net.use_cuda:
        x = sm.transform(x, 'NCHW2NHWC')
    x = sm.reshape(x, (-1, 16))
    conf, box, landmark = sm.split(x, (2, 6, 16), axis=-1)
    return conf, box, landmark


def onet_loss(conf, box, landmark, gt_conf, gt_box, gt_landmark, conf_mask, 
              box_mask, landmark_mask):
    """Create the loss of onet.
    
    Args:
        conf, box, landmark: The output of rnet.
        gt_conf, gt_box, gt_landmark: ground truth labels.
        conf_mask, box_mask, landmark_mask: to balance the three loss
    """
    conf_loss = conf_mask * sm.softmax_cross_entropy_with_logits(labels=gt_conf, 
                                                                 logits=conf)
    box_loss = box_mask * sm.hse(gt_box, box)
    landmark_loss = landmark_mask * sm.hse(gt_landmark, landmark)
    return conf_loss, box_loss, landmark_loss


def slim_conv2d(x, ci, co, filter_size, stride, padding, bias=True, 
                act=tf.nn.relu):
    weight = np.random.normal(0, 0.01, (co, ci, filter_size, filter_size))
    weight = np.transpose(weight, (2, 3, 1, 0))
    weight = tf.Variable(weight, dtype=tf.float32)
    y = tf.nn.conv2d(x, weight, (1, stride, stride, 1), padding=padding)
    if bias:
        bias = tf.Variable(np.zeros(shape=(co, )), dtype=tf.float32)
        y += bias
    if act:
        y = act(y)
    return y


def tf_pnet(x):
    """The same as `pnet`."""
    np.random.seed(196)

    x = slim_conv2d(x, 3, 10, 3, 1, padding='VALID')
    x = tf.nn.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME')
    x = slim_conv2d(x, 10, 16, 3, 1, padding='VALID')
    x = slim_conv2d(x, 16, 32, 3, 1, padding='VALID')
    x = slim_conv2d(x, 32, 16, 1, 1, padding='VALID', act=None) 
    conf, box, landmark = tf.split(x, (2, 4, 10), axis=-1)
    return conf, box, landmark


def tf_rnet(x):
    """The same as rnet."""
    np.random.seed(196)

    x = slim_conv2d(x, 3, 28, 3, 1, padding='VALID')
    x = tf.nn.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='SAME')
    x = slim_conv2d(x, 28, 48, 3, 1, padding='VALID')
    x = tf.nn.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding='VALID')
    x = slim_conv2d(x, 48, 64, 2, 1, padding='VALID')
    # use conv2d to replace full-connect
    x = slim_conv2d(x, 64, 128, 3, 1, padding='VALID')
    x = slim_conv2d(x, 128, 16, 1, 1, padding='VALID', act=None)
    x = tf.reshape(x, (-1, 16))
    conf, box, landmark = tf.split(x, (2, 4, 10), axis=-1)
    return conf, box, landmark


def tf_pnet_loss(conf, box, landmark, gt_conf, gt_box, gt_landmark, conf_mask, 
                 box_mask, landmark_mask):
    """The same as `pnet_loss`."""
    conf_loss = conf_mask * tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_conf, 
                                                                    logits=conf)
    box_loss = box_mask * 0.5 *  tf.square(gt_box - box)
    landmark_loss = landmark_mask * 0.5 * tf.square(gt_landmark - landmark)
    return conf_loss, box_loss, landmark_loss


def maer(a, b, gt):
    a = np.array(a).reshape(-1)
    b = np.array(b).reshape(-1)
    gt = np.array(gt).reshape(-1)
    keep = (a - b) != 0
    a = a[keep]
    b = b[keep]
    gt = gt[keep]
    if a.size == 0:
        return 0.
    smae = np.max(np.abs(a - b) / np.abs(gt))
    smae_id = np.argmax(np.abs(a - b) / np.abs(gt))
    if smae > thrs:
        print(a)
        print()
        print(b)
        print(a[smae_id])
        print(b[smae_id])
        pass
        #raise ValueError('Too large error in func self_max_abs_error: {}, '
        #                 'the threshold is {}'.format(smae, thrs))
    return smae


def check_rnet():
    import time
    lr = 1/32
    np.random.seed(196)

    x = sm.Tensor()
    gt_conf = sm.Tensor()
    gt_box = sm.Tensor() 
    gt_landmark = sm.Tensor() 
    conf_mask = sm.Tensor() 
    box_mask = sm.Tensor() 
    landmark_mask = sm.Tensor()
    conf, box, landmark = rnet(x)
    conf_loss, box_loss, landmark_loss = rnet_loss(conf, box, landmark, 
                                                   gt_conf, gt_box, 
                                                   gt_landmark, conf_mask, 
                                                   box_mask, landmark_mask)
    
    # check same shape
    epoch = 3
    np.random.seed(196)
    attrs = []
    for _ in range(epoch):
        h = 24
        w = 24
        dx = np.random.uniform(-1, 1, (32, 24, 24, 3))
        dconf = np.random.choice([0, 1], (32, 1))
        dconf = np.concatenate([dconf, 1-dconf], -1)
        dbox = np.random.randn(32, 4)
        dlandmark = np.random.randn(32, 10)
        dconf_mask = np.random.randn(32)
        dbox_mask = np.random.randn(32, 4)
        dlandmark_mask = np.random.randn(32, 10)

        conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_ = sm.forward(
            [conf, box, landmark, conf_loss, box_loss, landmark_loss],
            {x: dx, gt_conf: dconf, gt_box: dbox, gt_landmark: dlandmark,
            conf_mask: np.stack([dconf_mask] * 2, -1), box_mask: dbox_mask, landmark_mask: dlandmark_mask})
        sm.optimize([conf_loss, box_loss, landmark_loss], lr=lr)

        attrs.append([conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_])


    with tf.device('/cpu:0'):
        x1 = tf.placeholder(tf.float32)
        gt_conf1 = tf.placeholder(tf.float32)
        gt_box1 = tf.placeholder(tf.float32) 
        gt_landmark1 = tf.placeholder(tf.float32)
        conf_mask1 = tf.placeholder(tf.float32) 
        box_mask1 = tf.placeholder(tf.float32)
        landmark_mask1 = tf.placeholder(tf.float32)
        conf1, box1, landmark1 = tf_rnet(x1)
        conf_loss1, box_loss1, landmark_loss1 = tf_pnet_loss(conf1, box1, landmark1, 
                                                            gt_conf1, gt_box1, 
                                                            gt_landmark1, conf_mask1, 
                                                            box_mask1, landmark_mask1)
        loss = tf.reduce_sum(conf_loss1) + tf.reduce_sum(box_loss1) + tf.reduce_sum(landmark_loss1)
        opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    np.random.seed(196)
    epoch = 3
    attrs1 = []
    for _ in range(epoch):
        h = 24
        w = 24
        dx = np.random.uniform(-1, 1, (32, 24, 24, 3))
        dconf = np.random.choice([0, 1], (32, 1))
        dconf = np.concatenate([dconf, 1-dconf], -1)
        dbox = np.random.randn(32, 4)
        dlandmark = np.random.randn(32, 10)
        dconf_mask = np.random.randn(32)
        dbox_mask = np.random.randn(32, 4)
        dlandmark_mask = np.random.randn(32, 10)

        _, loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_ = sess.run(
            [opt, loss, conf1, box1, landmark1, conf_loss1, box_loss1, landmark_loss1],
            {x1: dx, gt_conf1: dconf, gt_box1: dbox, gt_landmark1: dlandmark,
            conf_mask1: dconf_mask, box_mask1: dbox_mask, landmark_mask1: dlandmark_mask})

        attrs1.append([loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_])

    print('RNet:')
    for (conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_), \
        (loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_) \
            in zip(attrs, attrs1):
        conf_error = maer(conf_, conf1_, conf1_)
        box_error = maer(box_, box1_, box1_)
        landmark_error = maer(landmark_, landmark1_, landmark1_)
        conf_loss_error = maer(np.sum(conf_loss_, -1), conf_loss1_, conf_loss1_)
        box_loss_error = maer(box_loss_, box_loss1_, box_loss1_)
        landmark_loss_error = maer(landmark_loss_, landmark_loss1_, landmark_loss1_)

        print('loss1_:', loss1_)
        print('loss_:', np.sum(conf_loss_) + np.sum(box_loss_) + np.sum(landmark_loss_))
        print('conf_error:', conf_error)
        print('box_error:', box_error)
        print('landmark_error:', landmark_error)
        print('conf_loss_error:', conf_loss_error)
        print('box_loss_error:', box_loss_error)
        print('landmark_loss_error:', landmark_loss_error)


if __name__ == '__main__':
    """Check the accuraty of pnet in smnet."""
    import time
    thrs = 1e-4
    lr = 1/32

    check_rnet()
    exit()
    np.random.seed(196)

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

    # check same shape
    epoch = 3
    np.random.seed(196)
    attrs = []
    for _ in range(epoch):
        #h = 224
        #w = 224
        h = np.random.randint(12, 224)
        w = np.random.randint(12, 224)
        subh = int(np.ceil((h - 2) / 2)) - 4
        subw = int(np.ceil((w - 2) / 2)) - 4
        dx = np.random.uniform(-1, 1, (32, h, w, 3))
        #dconf = np.random.randn(32, subh, subw, 2)
        dconf = np.random.choice([0, 1], (32, subh, subw, 1))
        dconf = np.concatenate([dconf, 1-dconf], -1)
        dbox = np.random.randn(32, subh, subw, 4)
        dlandmark = np.random.randn(32, subh, subw, 10)
        dconf_mask = np.random.randn(32, subh, subw)
        dbox_mask = np.random.randn(32, subh, subw, 4)
        dlandmark_mask = np.random.randn(32, subh, subw, 10)

        #DEBUG
        #dconf_mask *= 0
        #dbox_mask *= 0
        #dlandmark_mask *= 0

        conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_ = sm.forward(
            [conf, box, landmark, conf_loss, box_loss, landmark_loss],
            {x: dx, gt_conf: dconf, gt_box: dbox, gt_landmark: dlandmark,
            conf_mask: np.stack([dconf_mask] * 2, -1), box_mask: dbox_mask, landmark_mask: dlandmark_mask})
        sm.optimize([conf_loss, box_loss, landmark_loss], lr=lr)

        attrs.append([conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_])

    with tf.device('/cpu:0'):
        x1 = tf.placeholder(tf.float32)
        gt_conf1 = tf.placeholder(tf.float32)
        gt_box1 = tf.placeholder(tf.float32) 
        gt_landmark1 = tf.placeholder(tf.float32)
        conf_mask1 = tf.placeholder(tf.float32) 
        box_mask1 = tf.placeholder(tf.float32)
        landmark_mask1 = tf.placeholder(tf.float32)
        conf1, box1, landmark1 = tf_pnet(x1)
        conf_loss1, box_loss1, landmark_loss1 = tf_pnet_loss(conf1, box1, landmark1, 
                                                            gt_conf1, gt_box1, 
                                                            gt_landmark1, conf_mask1, 
                                                            box_mask1, landmark_mask1)
        loss = tf.reduce_sum(conf_loss1) + tf.reduce_sum(box_loss1) + tf.reduce_sum(landmark_loss1)
        opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    np.random.seed(196)
    epoch = 3
    attrs1 = []
    for _ in range(epoch):
        #h = 224
        #w = 224
        h = np.random.randint(12, 224)
        w = np.random.randint(12, 224)
        subh = int(np.ceil((h - 2) / 2)) - 4
        subw = int(np.ceil((w - 2) / 2)) - 4
        dx = np.random.uniform(-1, 1, (32, h, w, 3))
        dconf = np.random.choice([0, 1], (32, subh, subw, 1))
        dconf = np.concatenate([dconf, 1-dconf], -1)
        dbox = np.random.randn(32, subh, subw, 4)
        dlandmark = np.random.randn(32, subh, subw, 10)
        dconf_mask = np.random.randn(32, subh, subw)
        dbox_mask = np.random.randn(32, subh, subw, 4)
        dlandmark_mask = np.random.randn(32, subh, subw, 10)

        #DEBUG
        #dconf_mask *= 0
        #dbox_mask *= 0
        #dlandmark_mask *= 0

        _, loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_ = sess.run(
            [opt, loss, conf1, box1, landmark1, conf_loss1, box_loss1, landmark_loss1],
            {x1: dx, gt_conf1: dconf, gt_box1: dbox, gt_landmark1: dlandmark,
            conf_mask1: dconf_mask, box_mask1: dbox_mask, landmark_mask1: dlandmark_mask})

        attrs1.append([loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_])


    for (conf_, box_, landmark_, conf_loss_, box_loss_, landmark_loss_), \
        (loss1_, conf1_, box1_, landmark1_, conf_loss1_, box_loss1_, landmark_loss1_) \
            in zip(attrs, attrs1):
        conf_error = maer(conf_, conf1_, conf1_)
        box_error = maer(box_, box1_, box1_)
        landmark_error = maer(landmark_, landmark1_, landmark1_)
        conf_loss_error = maer(np.sum(conf_loss_, -1), conf_loss1_, conf_loss1_)
        box_loss_error = maer(box_loss_, box_loss1_, box_loss1_)
        landmark_loss_error = maer(landmark_loss_, landmark_loss1_, landmark_loss1_)

        print('loss1_:', loss1_)
        print('loss_:', np.sum(conf_loss_) + np.sum(box_loss_) + np.sum(landmark_loss_))
        print('conf_error:', conf_error)
        print('box_error:', box_error)
        print('landmark_error:', landmark_error)
        print('conf_loss_error:', conf_loss_error)
        print('box_loss_error:', box_loss_error)
        print('landmark_loss_error:', landmark_loss_error)

    epoch = 100
    h = 224
    w = 224
    subh = int(np.ceil((h - 2) / 2)) - 4
    subw = int(np.ceil((w - 2) / 2)) - 4
    dx = np.random.uniform(-1, 1, (32, h, w, 3))
    dconf = np.random.choice([0, 1], (32, subh, subw, 1))
    dconf = np.concatenate([dconf, 1-dconf], -1)
    dbox = np.random.randn(32, subh, subw, 4)
    dlandmark = np.random.randn(32, subh, subw, 10)
    dconf_mask = np.random.randn(32, subh, subw)
    dbox_mask = np.random.randn(32, subh, subw, 4)
    dlandmark_mask = np.random.randn(32, subh, subw, 10)
    # warmup
    for _ in range(10):
        sm.forward(
            [],
            {x: dx, gt_conf: dconf, gt_box: dbox, gt_landmark: dlandmark,
            conf_mask: np.stack([dconf_mask] * 2, -1), box_mask: dbox_mask, landmark_mask: dlandmark_mask})
        sm.optimize([conf_loss, box_loss, landmark_loss], lr=lr)
    t1 = time.time()
    for _ in range(epoch):
        sm.forward(
            [],
            {x: dx, gt_conf: dconf, gt_box: dbox, gt_landmark: dlandmark,
            conf_mask: np.stack([dconf_mask] * 2, -1), box_mask: dbox_mask, landmark_mask: dlandmark_mask})
        sm.optimize([conf_loss, box_loss, landmark_loss], lr=lr)
    t2 = time.time()
    print('sm train time: {}s'.format((t2 - t1) / epoch))

    """# warmup
    for _ in range(10):
        sess.run(
            [opt],
            {x1: dx, gt_conf1: dconf, gt_box1: dbox, gt_landmark1: dlandmark,
            conf_mask1: dconf_mask, box_mask1: dbox_mask, landmark_mask1: dlandmark_mask})
    t1 = time.time()
    for _ in range(epoch):
        sess.run(
            [opt],
            {x1: dx, gt_conf1: dconf, gt_box1: dbox, gt_landmark1: dlandmark,
            conf_mask1: dconf_mask, box_mask1: dbox_mask, landmark_mask1: dlandmark_mask})
    t2 = time.time()
    print('tf train time: {}s'.format((t2 - t1) / epoch))"""