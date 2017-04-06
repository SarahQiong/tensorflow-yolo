import sys
sys.path.append('./')
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from yolo.net.yolo_net import YoloNet
from yolo.dataset.text_dataset import TextDataSet
from demo import *
# in total we have 714 test images

class TextDataSetV2(TextDataSet):
    def __init__(self, common_params, dataset_params):
        super(TextDataSetV2, self).__init__(common_parameters, dataset_params)
    def record_process(self, record):
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.height, self.width))
        labels = [[0, 0, 0, 0, 0]] * self.max_objects
        i = 1
        object_num = 0
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            class_num = record[i + 4]
            labels[object_num] = [xmin, ymin, xmax, ymax, class_num]
            object_num += 1
            i += 5
            if object_num >= self.max_objects:
                break
        return [image, labels, object_num]

common_parameters = {'image_size': 448, 'batch_size': 16,\
    'num_classes': 2, 'max_objects_per_image': 10}
dataset_parameters = {'path': 'data/processed_data/test_label.txt', 'thread_num': 5}
dataset =  TextDataSetV2(common_parameters, dataset_parameters)
classes_name = ['bare soil', 'dead tree']
net_params = {'cell_size': 14, 'boxes_per_cell':2, 'weight_decay': 0.0005}
batch_size = common_parameters['batch_size']
net = YoloNet(common_parameters, net_params, test=True)
image = tf.placeholder(tf.float32, (None, 448, 448, 3))
predicts = net.inference(image)
sess = tf.Session()
saver = tf.train.Saver(net.trainable_collection)
ckpt = tf.train.get_checkpoint_state('models/train/full')
saver.restore(sess,ckpt.model_checkpoint_path)
num_imgs = 714 // batch_size * batch_size
predicted_boxes = np.zeros((num_imgs, net_params['cell_size'], net_params['cell_size'], \
    12), dtype=np.float32)
ground_truth_boxes = []
for i in range(714 // batch_size):
    images, labels, objects_num = dataset.batch()
    np_predict = sess.run(predicts, feed_dict={image: images})
    assert np_predict.shape[-1] == 12
    predicted_boxes[i * batch_size:(i+1) * batch_size] = np_predict
    labels = list(labels)
    for i in range(batch_size):
        temp = labels[i][:objects_num[i],:]
        ground_truth_boxes.append(temp[temp[:,4]==0]) # only consider boxes for dead tree class

def compute_precision_and_recall(predicted_boxes, ground_truth_boxes, p_threshold=0.1,\
        nms_threshold=0.5, true_iou_threshold=0.3):
    processed_boxes = []
    for i in range(len(predicted_boxes)):
        predicted_result = process_predicts(predicted_boxes[i], p_threshold)
        predicted_result = [tup for tup in predicted_result if tup[4]==0]
        nms_result = non_maximum_suppression(predicted_result, nms_threshold)
        processed_boxes.append(nms_result)
    # compute recall 
    r_tp = 0
    fn = 0
    for i in range(len(ground_truth_boxes)):
        for true_box in ground_truth_boxes[i]:
            fn += 1
            for predict_box in processed_boxes[i]:
                if iou(true_box[:4], predict_box[:4]) >= true_iou_threshold:
                    r_tp += 1
                    fn -= 1
                    break
    recall = r_tp / float(r_tp + fn)
    # compute precision
    p_tp = 0
    fp = 0
    for i in range(len(processed_boxes)):
        for predict_box in processed_boxes[i]:
            fp += 1
            for true_box in ground_truth_boxes[i]:
                if iou(true_box[:4], predict_box[:4]) >= true_iou_threshold:
                    p_tp += 1
                    fp -= 1
                    break
    if p_tp + fp > 0:
        precision = p_tp / float(p_tp + fp)
    else:
        precision = 1
    print('p_tp: {}\tr_tp: {}\tfn: {}\tfp: {}'.format(p_tp, r_tp, fn, fp))
    return precision, recall

thres_grid = np.arange(1e-5, 0.6, 1e-2)
precision_list = []
recall_list = []
for p_threshold in thres_grid:
    precision, recall = compute_precision_and_recall(predicted_boxes, ground_truth_boxes, p_threshold, 1, 0.3)
    precision_list.append(precision)
    recall_list.append(recall)
precision = np.asarray(precision_list)
recall = np.asarray(recall_list)
np.savez('precision_recall.npz', thres_grid, precision, recall)
keep = precision.argmax()
plt.plot(recall[:keep], precision[:keep])
plt.xlabel('recall rate')
plt.ylabel('precision rate')
plt.savefig('precision_recall_curve.png')
print('mAP: {}'.format(sum(np.diff(precision) * recall[:-1])))