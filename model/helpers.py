import os

import tensorflow as tf
import cv2
import numpy as np

def mtcnn_fun(img, min_size, factor, thresholds):
    with open(os.path.join(os.curdir, 'resources', 'mtcnn.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    print(box, prob, landmarks)
    return box, prob, landmarks

# wrap graph function as a callable function
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])

def detect(resp):
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    bbox, scores, landmarks = mtcnn_fun(image, 40, 0.7, [0.6, 0.7, 0.8])
    bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()
    res = []
    for box in bbox:
        d = dict()
        d['face'] = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
        res.append(d)
    return res
