import os
import sys
sys.path.append('./')
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

slim = tf.contrib.slim

VOC_LABELS = {
    0: 'none',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


net_shape = (300,300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE
)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(
        image_4d,
        is_training=False,
        reuse=reuse
    )
ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

ssd_anchors = ssd_net.anchors(net_shape)
print('ssd_anchors :', ssd_anchors)

def process_image(img,
                       select_threshold=0.5,
                       nms_threshold=.45,
                       net_shape=(300,300),):
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run(
        [image_4d, predictions, localisations, bbox_img],
        feed_dict={img_input: img}
    )
    print('rimg', rimg)
    print('rpredictions', rpredictions)
    print('rlocalisations', rlocalisations)
    print('rbbox_img', rbbox_img)


    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions,
        rlocalisations,
        ssd_anchors,
        select_threshold=select_threshold,
        img_shape=net_shape,
        num_classes=21,
        decode=True
    )
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(
        rclasses, rscores, rbboxes, top_k=400,
    )
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(
        rclasses, rscores, rbboxes, nms_threshold=nms_threshold,
    )
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    return rclasses, rscores, rbboxes


path = 'demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])
rclasses, rscores, rbboxes = process_image(img)

visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

#------------#
# add sample #
#------------#

colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255))
              for i in range(len(VOC_LABELS))]

def write_bboxes(img, classes, scores, bboxes):
    """ visualize bounding boxes, largely inspired by SSD-MXNET! """
    height = img.shape[0]
    width = img.shape[1]
    for i in range(classes.shapes[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], 2)
            class_name = VOC_LABELS[cls_id]
            cv2.rectangle(
                img, (xmin, ymin-6), (xmin+180, ymin+6), colors[cls_id], -1
            )
            cv2.putText(img,
                        '{:s} | {:.3f}'.format(class_name, score),
                        (xmin, ymin + 6),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255,255,255))



cap = cv2.VideoCapture('output.avi')
if not cap.isOpened():
    raise IOError(("couldn't open the video."))

capw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
caph = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'output_ssd.avi',
    int(fourcc),
    fps,
    (int(capw), int(caph)),
)

prev_time = time.time()
frame_cnt = 0
while True:
    ret, img = cap.read()
    if not ret:
        print('Done!')
        break

    rclasses, rscores, rbboxes = process_image(img)
    write_bboxes(img, rclasses, rscores, rbboxes)
    out.write(img)
    frame_cnt += 1

curr_time = time.time()
exec_time = curr_time - prev_time
print('FPS: {0}'.format(frame_cnt/exec_time))

cap.release()
out.release()
cv2.destroyAllWindows()
