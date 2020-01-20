#!/usr/bin/env python
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolo3_tf2_ros.msg import SSD_Output,SSD_Outputs
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import rospkg 

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def img_callback (ros_img):
    global bridge, cv_img
    try: 
        cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
    except CvBridgeError as e:
        print (e)

rospy.init_node('ssd_node', anonymous = True)
bridge = CvBridge()
img_sub = rospy.Subscriber('/camera/image_raw', Image, img_callback)
cv_img = rospy.wait_for_message('/camera/image_raw',Image)

rospack = rospkg.RosPack()
pa = rospack.get_path('yolo3_tf2_ros')


if __name__ == '__main__':
    
    
    
    Im_outs_Pub = rospy.Publisher('im_info',SSD_Outputs)
    img_pub = rospy.Publisher('ssd_image_output',Image)
    rate = rospy.Rate(5)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    yolo = YoloV3(classes=80)

    yolo.load_weights(pa+'/checkpoints/yolov3.tf').expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(pa+'/data/coco.names').readlines()]
    print('classes loaded')
    
    
    while not rospy.is_shutdown():
        L_output = SSD_Outputs()
        flag = 0
        img_raw = cv_img
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, 416)
        height = img.shape[0]
        width = img.shape[1]
        boxes, scores, classes, nums, probabilities = yolo(img)
        np_boxes = boxes.numpy()
        for i in range(nums):
            output = SSD_Output()
            ymin = np.int32(np_boxes[0][i][0] * height)
            xmin = np.int32(np_boxes[0][i][1] * width)
            ymax = np.int32(np_boxes[0][i][2] * height)
            xmax = np.int32(np_boxes[0][i][3] * width)
            output.x_min = int(xmin)
            output.x_max = int(xmax)
            output.y_min = int(ymin)
            output.y_max = int(ymax)
            output.height_factor = (ymax-ymin)/(xmax-xmin)
            output.cls = np.array(classes[0][i]).astype(np.int32)
            output.height_factor = (ymax-ymin)/(xmax-xmin)

            for p in probabilities[0][i]:
                output.probability_distribution.append(p)

            L_output.outputs.append(output)

        Im_outs_Pub.publish(L_output)
        img = draw_outputs(img_raw, (boxes, scores, classes, nums), class_names)
        msg_frame = bridge.cv2_to_imgmsg(img)
        msg_frame.encoding = 'bgr8'
        msg_frame.header.frame_id = 'camera_link'
        img_pub.publish(msg_frame)
        rate.sleep()



    
  

    
