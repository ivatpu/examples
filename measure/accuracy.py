"""Measure accuracy of a network on imagenet dataset."""
import os
import logging
from typing import Tuple
import numpy as np
from PIL import Image
import iva_applications
from iva_applications.utils import Runner
from iva_applications.imagenet.postprocess import tpu_tensor_to_num_classes
from iva_applications.preprocess_factory import PREPROCESS_FN
from iva_applications.offset_factory import OFFSETS

logger = logging.getLogger("Accuracy")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


def catch_hit(out_tensor: np.array, label: int) -> tuple:
    """Catch hit in output tensor."""
    hit_top_1, hit_top_5 = None, None
    if label in out_tensor[:5]:
        hit_top_5 = True
        if label == out_tensor[0]:
            hit_top_1 = True
    else:
        hit_top_1 = hit_top_5 = False
    return hit_top_1, hit_top_5


def val_file_to_dict(val_file_path: str, network_name: str) -> dict:
    """Preprocess test file to test dict."""
    val_dict = {}
    read_file = open(val_file_path, 'r')
    for string in read_file:
        image_path, label = string.split(' ')
        label = np.int16(label)
        if network_name not in OFFSETS:
            print('Available offsets:', OFFSETS.keys())
            raise ValueError('No offset value provided')
        label = label - OFFSETS[network_name]
        val_dict[image_path] = label
    return val_dict


def feed_imagenet(val_dict: dict, dataset_dir: str, network_name: str, runner, log_step: int):
    """Feed images for TPU/TF network."""
    image_counter = 0.0
    top_1_counter = 0.0
    top_5_counter = 0.0
    top_1 = 0.0
    top_5 = 0.0
    net_name = network_name.lower()
    if net_name not in PREPROCESS_FN:
        print('Supported nets:', PREPROCESS_FN.keys())
        raise ValueError('Test for network [%s] is not supported' % net_name)
    for image_path in val_dict.keys():
        image_counter += 1.0
        image = Image.open(os.path.join(dataset_dir, image_path))
        tensor = iva_applications.PREPROCESS_FN[net_name].preprocess.image_to_tensor(image)
        tensor = np.expand_dims(tensor, axis=0)
        output = runner({runner.input_nodes[0]: tensor})
        classes = tpu_tensor_to_num_classes(output[runner.output_nodes[0]], 5)
        label = val_dict[image_path]
        hit_top_1, hit_top_5 = catch_hit(classes, label)
        if hit_top_5:
            top_5_counter += 1.0
            if hit_top_1:
                top_1_counter += 1.0
        top_1 = top_1_counter / image_counter
        top_5 = top_5_counter / image_counter
        if image_counter % log_step == 0:
            logger.info(' %s    top1 = %.05f, top5 = %.05f,   step = %s/%s', net_name, top_1, top_5, int(image_counter),
                        len(val_dict.keys()))
    return top_1, top_5


def evaluate_network(val_file_path: str, dataset_dir: str, network_name: str, runner: Runner, log_step: int) -> \
        Tuple[float, float]:
    """
    Measure top1 and top5 accuracy of a network on Imagenet dataset.

    :param val_file_path: path to the imagenet validation file
    :param dataset_dir: path to the imagenet dataset directory
    :param network_name: name of the network from PREPROCESS_FN
    :param runner: TF or TPU runner used to run the network
    :param log_step: write current accuracy every log_step steps
    :return: top1 and top5 accuracy for measured network
    """
    val_dict = val_file_to_dict(val_file_path, network_name)
    top_1, top_5 = feed_imagenet(val_dict, dataset_dir, network_name, runner, log_step)
    return top_1, top_5
