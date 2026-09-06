from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from joblib import Parallel, delayed
from isegm.inference.utils import setup_deterministic

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm



def evaluate_functor(dataset, predictor, index, **kwargs):

    if 'support_deterministic' in kwargs and kwargs['support_deterministic']:
        setup_deterministic(seed=0)
    else:
        setup_deterministic(seed=0, is_full=False)

    all_ious = []
    sample = dataset.get_sample(index)
    for obj_id in sample.objects_ids:
        gt_mask = sample.get_object_mask(obj_id)
        _, sample_ious, _ = evaluate_sample(sample.image, gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append([sample.imname, sample_ious, sample.objects_ids])
    return all_ious


def evaluate_dataset(dataset, predictor, **kwargs):

    all_ious = []
    start_time = time()
    dataset_iterator = tqdm(range(len(dataset)), leave=False)

    if kwargs['args'].n_workers == 1:
        for index in dataset_iterator:
            all_ious.append(evaluate_functor(dataset, predictor, index, **kwargs))
    else:
        all_ious = Parallel(n_jobs=kwargs['args'].n_workers)(delayed(evaluate_functor)(dataset, predictor, index, **kwargs) for index in dataset_iterator)

    # Flatten list of lists
    all_ious = [item for sublist in all_ious for item in sublist]
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time



def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, args=None, support_deterministic=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    predictor.set_input_image(image)

    for click_indx in range(max_clicks):
        clicker.make_next_click(pred_mask)

        pred_probs, metrics_dict = predictor.get_prediction(clicker, args=args)

        with torch.no_grad():
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            biou = utils.get_boundary_iou(gt_mask, pred_mask)

            ious_list.append([iou, biou, metrics_dict])

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

    return clicker.clicks_list, np.array(ious_list), pred_probs
