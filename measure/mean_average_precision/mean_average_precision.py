"""Measure mean Average Precision metric for YOLO3."""
import logging
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
from .coco import Annotator
from .map_utils import calculate_map
from iva_applications.mscoco17.config import CLASS_NAMES
from iva_applications.utils import Runner
from iva_applications import yolo2

logger = logging.getLogger("Mean average precision")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


def filter_detections(detections: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Pop empty classes and convert to Ground Truth format.

    Parameters
    ----------
    detections
        Postprocessing tensors result

    Returns
    -------
    Postprocessed tensors result without empty classes
    """
    labels = []
    labels_which_detect = [None if detections[key].size == 0 else key for key in list(detections.keys())]
    for value in labels_which_detect:
        if value is not None:
            number_of_cl_detect = np.shape(detections[value][:, :5])[0]
            for num in range(number_of_cl_detect):
                labels.append((value, *detections[value][num, :5]))
    detections_as_array = np.asarray(labels)
    return detections_as_array


def rescale_post(postprocessed: np.ndarray, size: tuple, yolo_size: int) -> np.ndarray:
    """
    Rescale image according to the image size.

    Parameters
    ----------
    postprocessed
        initial postprocessed data
    size
        size of the image
    yolo_size
        size of the network (YOLO3) input

    Returns
    -------
    np.ndarray of resized boxes
    """
    for box in range(np.shape(postprocessed)[0]):
        class_, xmin, ymin, xmax, ymax, confidence = postprocessed[box]
        ymin = ymin * size[1] / yolo_size
        xmin = xmin * size[0] / yolo_size
        ymax = ymax * size[1] / yolo_size
        xmax = xmax * size[0] / yolo_size
        postprocessed[box] = class_, xmin, ymin, xmax, ymax, confidence
    return postprocessed


def get_network_outputs(images_path: str, instances_path: str, runner: Runner,
                        input_tensor_names: list, conv_outputs_names: list, postprocessing_graph: tf.Graph,
                        yolo_size: int, log_step: int = 1000) -> Tuple[List, List, List]:
    """
    Obtain the outputs of the network.

    Parameters
    ----------
    images_path
        path to the images
    instances_path
        path to the instances file
    runner
        runner used to run the network
    input_tensor_names
        names of the input tensors
    conv_outputs_names
        names of the outputs of the main graph (2 outputs for Tiny-YOLO3, 3 outputs for YOLO3)
    postprocessing_graph
        postprocessing graph used to construct boxes from the main output
    yolo_size
        size of the network (YOLO3 or Tiny-YOLO3) input
    log_step
        log progress every log_step images

    Returns
    -------
    Tuple of lists:
                list of postprocessed network outputs for each image
                list of ground truths for each image
                list of corresponding image names
    """
    annotator = Annotator(instances_path, images_path, CLASS_NAMES)
    dataset_generator = annotator.generator()
    images_list = annotator.list_images()

    prediction_list = []
    ground_truth_list = []
    image_list = []
    image_counter = 0

    for image_path, ground_truth in dataset_generator:
        image = Image.open(image_path)
        tensor = yolo2.preprocess.image_to_tensor(image, yolo_size)
        tensor = np.expand_dims(tensor, 0)
        output_tensors = runner({input_tensor_names[0]: tensor})
        with tf.compat.v1.Session() as sess:
            if len(conv_outputs_names) == 2:
                postprocessed = sess.run(postprocessing_graph, feed_dict={
                    sess.graph.get_tensor_by_name(conv_outputs_names[0]): output_tensors[conv_outputs_names[0]],
                    sess.graph.get_tensor_by_name(conv_outputs_names[1]): output_tensors[conv_outputs_names[1]]})
            elif len(conv_outputs_names) == 3:
                postprocessed = sess.run(postprocessing_graph, feed_dict={
                    sess.graph.get_tensor_by_name(conv_outputs_names[0]): output_tensors[conv_outputs_names[0]],
                    sess.graph.get_tensor_by_name(conv_outputs_names[1]): output_tensors[conv_outputs_names[1]],
                    sess.graph.get_tensor_by_name(conv_outputs_names[2]): output_tensors[conv_outputs_names[2]]})
            else:
                raise ValueError('Number of output tensors is incorrect (Got %d, expected 2 or 3).' %
                                 len(conv_outputs_names))
        image_counter += 1

        if len(postprocessed) != 0:
            postprocessed_reformat = filter_detections(postprocessed[0])
            rescaled_postprocessed = rescale_post(postprocessed_reformat, image.size, yolo_size)
            ground_truth = np.asarray(ground_truth)
            prediction_list.append(rescaled_postprocessed)
            ground_truth_list.append(ground_truth)
            image_list.append(image_path)

        if image_counter % log_step == 0:
            logger.info('step = %s/%s', int(image_counter), len(images_list))
            mean_ap = calculate_map(instances_path, prediction_list, image_list)
            logger.info('mAP at %s images is %s', int(image_counter), mean_ap)
    return prediction_list, ground_truth_list, image_list
