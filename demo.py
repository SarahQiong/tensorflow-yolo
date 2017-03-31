import sys

sys.path.append('./')

from yolo.net.yolo_net import YoloNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ['bare soil', 'dead tree']

def iou(boxes1, boxes2):
    lu = np.maximum(boxes1[0:2], boxes2[0:2])
    rd = np.minimum(boxes1[2:], boxes2[2:])
    intersection = np.maximum(rd - lu, [0,0])
    inter_square = intersection[0]*intersection[1]
    square1 = (boxes1[2]-boxes1[0]) * (boxes1[3] - boxes1[1])
    square2 = (boxes2[2]-boxes2[0]) * (boxes2[3] - boxes2[1])
    return inter_square / (square1 + square2 + 1e-5 - inter_square)

def process_predicts(predicts, p_threshold=0.01):
    result = []
    if len(predicts.shape) == 3:
        predicts = predicts[np.newaxis, :]
    p_classes = predicts[0, :, :, 0:2]
    C = predicts[0, :, :, 2:4]
    coordinate = predicts[0, :, :, 4:]
    coordinate = np.reshape(coordinate, (7, 7, 2, 4))
    p_classes = np.reshape(p_classes, (7, 7, 1, 2))
    C = np.reshape(C, (7, 7, 2, 1))

    P = C * p_classes

    index = np.argmax(P)

    index = np.unravel_index(index, P.shape)
    while(P[index] > p_threshold):
        class_num = index[3]
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
        xcenter = (index[1] + xcenter) * (448/7.0)
        ycenter = (index[0] + ycenter) * (448/7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h
        result.append((xmin, ymin, xmax, ymax, class_num, P[index]))
        
        P[index]=0
        index = np.argmax(P)
        index = np.unravel_index(index, P.shape)
    return result

def _one_step_suppression(box_list, iou_threshold):
    remove_list = []
    for box in box_list[1:]:
        if iou(box_list[0][:4], box[:4]) >= iou_threshold:
            remove_list.append(box)
    box_list = [b for b in box_list if b not in remove_list]
    return box_list

def non_maximum_suppression(box_list, iou_threshold):
    if len(box_list) <= 1:
        return box_list
    output = []
    while(True):
        box_list = _one_step_suppression(box_list, iou_threshold)
        assert len(box_list) > 0
        if len(box_list) > 1:
            output.append(box_list[0])
            box_list = box_list[1:]
        else:
            output.append(box_list[0])
            return output

if __name__ == "__main__":
    common_params = {'image_size': 448, 'num_classes': 2, 
                                    'batch_size':1}
    net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

    net = YoloNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (1, 448, 448, 3))
    predicts = net.inference(image)

    sess = tf.Session()

    np_img = cv2.imread('/home/rui/tensorflow-yolo/data/processed_data/test/D1019020201_ob77_crop0.JPG')
    resized_img = cv2.resize(np_img, (448, 448))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    saver = tf.train.Saver(net.trainable_collection)
    ckpt = tf.train.get_checkpoint_state('models/train/backup_full_model')
    saver.restore(sess,ckpt.model_checkpoint_path)

    np_predict = sess.run(predicts, feed_dict={image: np_img})

    box_list = process_predicts(np_predict, 0.04)
    box_list = non_maximum_suppression(box_list, 0.5)
    for ob in box_list:
        xmin, ymin, xmax, ymax, class_num, p = ob
        class_name = classes_name[class_num]
        cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
        cv2.putText(resized_img, class_name + ' {:.2f}'.format(p), (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
    cv2.imwrite('test_out.jpg', resized_img)
    sess.close()
