from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

from PIL import Image


class TLClassifier(object):
    """
    This class uses the trained model from the TensorFlow Object Detection API.
    https://github.com/tensorflow/models/tree/master/research/object_detection

    For the Capstone Project, a Single Shot Detector with lightweight MobileNet
    Backbone was trained on the Bosch Traffic Light Dataset, LISA Traffic Light Dataset and
    Udacity Simulator and Site images.

    The Inference Code was adapted from:
    https://github.com/tensorflow/models/blob/master/research/object_detection/inference/detection_inference.py

    """
    def __init__(self, path_to_tensorflow_graph, thresholds):

        # Create the TensorFlow session in which the graph is loaded
        self.session = tf.Session()
        self.thresholds = thresholds

        with self.session.as_default():
            saver = tf.train.import_meta_graph(path_to_tensorflow_graph+'/model.ckpt.meta')
            saver.restore(self.session, tf.train.latest_checkpoint(path_to_tensorflow_graph))

            # Create Tensors for results
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_name_keras_phase = [x for x in all_tensor_names if 'keras_learning_phase' in x][0]

            self.activation_map_tensor = tf.get_default_graph().get_tensor_by_name('conv2d_6/BiasAdd:0')
            self.activation_map_softmax = tf.nn.softmax(self.activation_map_tensor)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('input_1:0')
            self.learning_phase_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name_keras_phase)

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image by
        using the TensorFlow graph.

        We run the operations that will give us the boxes, scores and labels.
        Then we filter out the most probable scores (> threshold) and use the
        biggest box, since this will be the nearest traffic light.

        The graph will give us the following IDs :
        3: yellow
        2: red
        1: none
        0: green

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Threshold for detections
        self.class_threshs = {TrafficLight.GREEN: self.thresholds[0],
                              TrafficLight.YELLOW: self.thresholds[1],
                              TrafficLight.RED: self.thresholds[2]}

        self.class_indices = {0: TrafficLight.GREEN,
                              # 1: 'none',
                              2: TrafficLight.RED,
                              3: TrafficLight.YELLOW}

        traffic_light_id = 4  # 4 equals to unknown

        results = self.session.run({'activation_map_softmax' : self.activation_map_softmax },
                                   feed_dict={self.image_tensor: image,
                                              self.learning_phase_tensor: False})

        results_aug = self.session.run({'activation_map_softmax' : self.activation_map_softmax },
                                   feed_dict={self.image_tensor: np.expand_dims(np.fliplr(image[0, :, :, :]), 0),
                                              self.learning_phase_tensor: False})


        activation_map = results['activation_map_softmax'][0, :, :, :]
        activation_map_aug = results_aug['activation_map_softmax'][0, :, :, :]
        activation_map = (activation_map + np.fliplr(activation_map_aug)) / 2.0

        class_probabilities = {}
        [class_probabilities.update({self.class_indices[i]: np.max(activation_map[:, :, i])}) for i in self.class_indices]

        class_prediction = int(max(class_probabilities, key=class_probabilities.get))
        class_prob = class_probabilities[class_prediction]
        class_prediction = class_prediction if class_prob > self.class_threshs[class_prediction] else TrafficLight.UNKNOWN

        return class_prediction


class TestTLClassifier(object):

    def __init__(self):
        self.detector = TLClassifier()

    def test_classification(self):
        # Load image
        image_path_green = ('light_classification/test_images/green.jpg', TrafficLight.GREEN)
        image_path_yellow = ('light_classification/test_images/yellow.jpg', TrafficLight.YELLOW)
        image_path_red = ('light_classification/test_images/red.jpg', TrafficLight.RED)
        image_path_na = ('light_classification/test_images/NA.jpg', TrafficLight.UNKNOWN)

        for image_path in [image_path_green, image_path_yellow, image_path_red, image_path_na]:
            image = np.asarray(Image.open(image_path[0]))
            image = np.expand_dims(image, 0)
            gt_result = image_path[1]
            pred_result = self.detector.get_classification(image)
            print(image_path[0])
            print('Prediction success: ' + str(gt_result == pred_result))

            if gt_result != pred_result:
                raise Exception('Prediction error.')

    def measure_time(self):
        # Load image
        image_path = 'light_classification/test_images/green.jpg'
        image = np.asarray(Image.open(image_path))
        image = np.expand_dims(image, 0)

        repeats = 25

        t0 = datetime.datetime.now()
        for i in range(repeats):
            _ = self.detector.get_classification(image)

        delta = datetime.datetime.now() - t0
        print('Time per image in ms: ' + str(delta.seconds * 100.0 / float(repeats)))


if __name__ == '__main__':
    tester = TestTLClassifier()
    tester.measure_time()
    tester.test_classification()
