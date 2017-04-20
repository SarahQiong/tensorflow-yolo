import sys
sys.path.append('./')
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from yolo.net.yolo_net import YoloNet
from demo import *
IMG_FILE = '/home/rui/tensorflow-yolo/data/AerialTree/D1019020236.JPG'
OUTPUT_PATH = '/home/rui/tensorflow-yolo/data/'
WINDOW_SIZE = 448
OVERLAP = 100
# suppose height and width of input image are greater than WINDOW_SIZE
np_img = cv2.imread(IMG_FILE)
np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
print(np_img.shape)
height, width, _ = np_img.shape
x_list = [(WINDOW_SIZE - OVERLAP) * x for x in range((width - OVERLAP) // (WINDOW_SIZE - OVERLAP))] + \
    [width - WINDOW_SIZE]
y_list = [(WINDOW_SIZE - OVERLAP) * y for y in range((height - OVERLAP) // (WINDOW_SIZE - OVERLAP))] + \
    [height - WINDOW_SIZE]

# load model
common_parameters = {'image_size': 448, 'batch_size': 16,\
    'num_classes': 2, 'max_objects_per_image': 10}
classes_name = ['bare soil', 'dead tree']
net_params = {'cell_size': 14, 'boxes_per_cell':2, 'weight_decay': 0.0005}
batch_size = common_parameters['batch_size']
net = YoloNet(common_parameters, net_params, test=True)
image_placeholder = tf.placeholder(tf.float32, (None, 448, 448, 3))
predicts = net.inference(image_placeholder)
sess = tf.Session()
saver = tf.train.Saver(net.trainable_collection)
ckpt = tf.train.get_checkpoint_state('models/train/full')
saver.restore(sess,ckpt.model_checkpoint_path)

crop_list = []
for x in x_list:
    for y in y_list:
        image = np.asarray(np_img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE], dtype=np.float32)
        image = image/255 * 2 - 1
        crop_list.append(((x, y), image))

def process_predicts(predicts, ul_cordinate, keep_class=1, p_threshold=0.1):
    #accept one input
    if type(keep_class) is not list:
        keep_class = [keep_class]
    result = []
    if len(predicts.shape) == 3:
        predicts = predicts[np.newaxis, :]
    p_classes = predicts[0, :, :, 0:2]
    C = predicts[0, :, :, 2:4]
    coordinate = predicts[0, :, :, 4:]
    coordinate = np.reshape(coordinate, (14, 14, 2, 4))
    p_classes = np.reshape(p_classes, (14, 14, 1, 2))
    C = np.reshape(C, (14, 14, 2, 1))

    P = C * p_classes

    index = np.argmax(P)

    index = np.unravel_index(index, P.shape)
    while(P[index] > p_threshold):
        class_num = index[3]
        if class_num not in keep_class:
            P[index]=0
            index = np.argmax(P)
            index = np.unravel_index(index, P.shape)
            continue
        max_coordinate = coordinate[index[0], index[1], index[2], :]
        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]
        if w < 0 or h < 0:
            P[index]=0
            index = np.argmax(P)
            index = np.unravel_index(index, P.shape)
            continue
        xcenter = (index[1] + xcenter) * (448/14.0)
        ycenter = (index[0] + ycenter) * (448/14.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w/2.0 + ul_cordinate[0]
        ymin = ycenter - h/2.0 + ul_cordinate[1]

        xmax = xmin + w
        ymax = ymin + h
        result.append((xmin, ymin, xmax, ymax, class_num, P[index]))
        
        P[index]=0
        index = np.argmax(P)
        index = np.unravel_index(index, P.shape)
    return result

num_crops = len(crop_list)
predicted_boxes = np.zeros((num_crops, net_params['cell_size'], net_params['cell_size'], \
    12), dtype=np.float32)
for i in range(num_crops//batch_size):
    input_imgs = np.stack([tup[1] for tup in crop_list[i*batch_size:(i+1)*batch_size]], axis=0)
    np_predict = sess.run(predicts, feed_dict={image_placeholder: input_imgs})
    predicted_boxes[i*batch_size:(i+1)*batch_size] = np_predict
if (i+1) * batch_size < num_crops:
    input_imgs = np.stack([tup[1] for tup in crop_list[(i+1)*batch_size:]], axis=0)
    np_predict = sess.run(predicts, feed_dict={image_placeholder: input_imgs})
    predicted_boxes[(i+1)*batch_size:] = np_predict
# draw bounding box on original image
bounding_box_list = []
for i in range(num_crops):
    bounding_box_list.extend(process_predicts(predicted_boxes[i,:,:,:], crop_list[i][0]))
bounding_box_list.sort(key=lambda x:x[5], reverse=True)
bounding_box_list = non_maximum_suppression(bounding_box_list, 0.5)
#notate the image
np_img = cv2.imread(IMG_FILE)
for box in bounding_box_list:
    cv2.rectangle(np_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
cv2.imwrite(OUTPUT_PATH + 'test_full_img.JPG', np_img)