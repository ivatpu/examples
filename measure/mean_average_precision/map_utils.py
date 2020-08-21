"""
Utils used for mean average precision (mAP) calculation.

Calculate
tp - true positive,
tn - true negative,
fp - false positive,
fn - false negative
predictions

Then calculate the Precision and the Recall
"""
import os
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from iva_applications.ssd_mobilenet.postprocess import xyxy2xywh, coco80_to_coco91_class


def iou_measure(detection: Tuple[int, float, float, float, float],
                ground_truth: Tuple[int, float, float, float, float]) -> float:
    """
    Calculate intersection over union with ground truth.

    Boxes coordinates are given in order (xmin, ymin, xmax, ymax)
    :param detection: predicted data after postprocessing
    :param ground_truth: true data after postprocessing
    :return: overlap
    """
    box = detection[1:]
    truth_box = ground_truth[1:]
    overlap = 0.
    box_inter = [
        np.maximum(box[0], truth_box[0]),
        np.maximum(box[1], truth_box[1]),
        np.minimum(box[2], truth_box[2]),
        np.minimum(box[3], truth_box[3])
    ]
    horz_inter = box_inter[3] - box_inter[1]
    vert_inter = box_inter[2] - box_inter[0]
    if horz_inter > 0 and vert_inter > 0:
        detection_area = (box[3] - box[1]) * (box[2] - box[0])
        truth_area = (truth_box[3] - truth_box[1]) * (truth_box[2] - truth_box[0])
        union_area = detection_area + truth_area - horz_inter * vert_inter
        overlap = horz_inter * vert_inter / union_area
    return overlap


def sort_predictions(predictions: list, gt_list: list) -> Tuple[List, List]:
    """
    Sort the predictions and corresponding ground truth values by the scores in descending order.

    :param predictions: The list of all predictions
    :param gt_list: The list of corresponding ground truth values
    :return: The tuple of lists of predictions and ground truth values
    """
    pr_sorted = []
    gt_sorted = []
    scores = []
    for prediction in predictions:
        score = prediction[:, -1]
        scores.append(max(score))
    inds = np.argsort(scores)
    inds = inds[::-1]
    for i, _ in enumerate(inds):
        sorted_index = inds[i]
        pr_sorted.append(predictions[sorted_index])
        gt_sorted.append(gt_list[sorted_index])
    return pr_sorted, gt_sorted


def voc_ap(rec: np.ndarray, prec: np.ndarray, iou_thresh: float) -> float:
    """
    Calculate Average Precision (AP) using given precision and recall values and IoU threshold value.

    This function calculates the area under the precision-recall graph.
    :param rec: recall values for the objects
    :param prec: precision values for the objects
    :param iou_thresh: given IoU threshold for which AP in calculated
    :return: AP value
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under the precision-recall curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    plot_pr_curve(mrec, mpre, iou_thresh)
    # and sum (\Delta recall) * precision
    aver_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return aver_precision


def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, iou_thresh: float) -> None:
    """
    Plot and save the precision-recall curve with the corresponding IoU threshold.

    :param recall: array of recall values
    :param precision: array of precision values
    :param iou_thresh: given IoU threshold
    :return: None
    """
    # pylint: disable=import-outside-toplevel
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    name = 'pr_rc' + str(int(iou_thresh * 100)) + '.png'
    fig.savefig(name, dpi=fig.dpi)


def make_results_list(prediction_list: list, images_list: list) -> list:
    """
    Conctruct a list of dicts for pycocotools.

    This function takes predictions and corresponding images paths and writes them
    into the list of dicts in format suitable for pycocotools.
    Construct a list of dictionaries containing the output of the network in form
    [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...]

    :param prediction_list: list of predictions [[class, xmin, ymin, xmax, ymax, score],...]
    :param images_list: list of paths of the images
    :return: list of dicts of format:
    {
            'image_id': id of the given image,
            'category_id': id of predicted class,
            'bbox': bounding box in format [xmin, ymin, width, height],
            'score': confidence score
    }
    """
    coco91class = coco80_to_coco91_class()
    result = []
    for _, (predictions, img_path) in enumerate(zip(prediction_list, images_list)):
        img_id = Path(img_path).stem
        for _, predict in enumerate(predictions):
            class_ = int(predict[0])
            box = predict[1:5]
            box = xyxy2xywh(box)  # xywh
            box[:2] -= box[2:] / 2  # xy center to top-left corner
            box = box.tolist()
            score = predict[-1]
            image_id = int(img_id)
            category_id = coco91class[class_]
            result.append({
                'image_id': image_id,
                'category_id': category_id,
                'bbox': box,
                'score': score
            })
    return result


def calculate_map(instances_path: str, prediction_list: list, images_list: list) -> float:
    """
    Take predictions of the network and calculate mAP for COCO2017 dataset.

    The function takes
    :param instances_path: path to the instances file
    :param prediction_list: list of postprocessed network outputs for each image
    :param images_list: list of corresponding image names
    :return: mAP value
    """
    result_list = make_results_list(prediction_list, images_list)
    img_ids = [int(Path(img_path).stem) for img_path in images_list]
    results_dump_file = 'results.json'
    with open(results_dump_file, 'w') as file:
        json.dump(result_list, file)

    coco_ground_truth = COCO(instances_path)
    detections = coco_ground_truth.loadRes(results_dump_file)
    os.remove(results_dump_file)
    coco_evaluation = COCOeval(coco_ground_truth, detections, 'bbox')

    coco_evaluation.params.imgIds = img_ids
    coco_evaluation.evaluate()
    coco_evaluation.accumulate()
    coco_evaluation.summarize()
    mean_ap = coco_evaluation.stats[0]  # stats[0] records AP@[0.5:0.95]
    print("mAP: {}".format(coco_evaluation.stats[0]))
    return mean_ap


def match_quant_original_nodes(conf_tpu_json_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Make a dict with input and output quant nodes names as keys and corresponding original nodes names as values.

    This function uses a dict from to_compile/conf.tpu.json which is obtained after quantization.

    Parameters
    ----------
    conf_tpu_json_dict
        dict from conf.tpu.json file

    Returns
    -------
    The dict which matches nodes of quantized graph and original nodes
    """
    match_dict = {}
    in_out_nodes = []
    comp_dict = conf_tpu_json_dict['model']
    all_nodes = list(comp_dict.keys())
    in_out_raw_nodes = [node for node in all_nodes if 'input_node' in node or 'output_node' in node]
    for node in in_out_raw_nodes:
        if 'input_node' in node:
            node = node + '/Placeholder:0'
        if 'output_node' in node:
            node = node + '/output:0'
        in_out_nodes.append(node)
    for node in in_out_nodes:
        match_dict[node] = comp_dict[node.split('/')[0]]['anchor']
    return match_dict
