import sys
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.datasets import parse_dataset_names
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
from isegm.inference.evaluation import evaluate_dataset
import os
import random
from isegm.inference.transforms.zoom_in_gpcis import ZoomIn
from isegm.inference.predictors.gpcis import BaselinePredictor
from evaluate_model_ritm import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback
from evaluate_model_ritm import get_predictor_and_zoomin_params, parse_args
from isegm.inference.utils import setup_deterministic


def parse_args():
    parser = argparse.ArgumentParser(description='Оценка устойчивости интерактивной сегментации к положению кликов.')

    parser.add_argument('mode', choices=[ 'CDNet', 'Baseline', 'FocalClick', 'NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='Режим предсказания маски')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='Путь к файлу весов. '
                                        'Допускается относительный путь от cfg.INTERACTIVE_MODELS_PATH '
                                        'или абсолютный путь. Расширение файла можно опустить.')

    parser.add_argument('--model_dir', type=str, default='',
                                   help='Путь к файлу весов.')

    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='Относительный путь к эксперименту с весами. '
                                        '(относительно cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=parse_dataset_names, default='GrabCut,Berkeley,DAVIS,COCO-MVal',
                        help='Датасеты для оценки модели. '
                             'Укажите имена через запятую. Доступны: '
                             'GrabCut, Berkeley, DAVIS, COCO-MVal')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='Идентификаторы используемых видеокарт.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Выполнять предсказания только на процессоре.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Целевой порог IoU для метрики NoC (минимум 0.8).')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Сохранить данные зависимости mIoU от числа кликов для полного числа взаимодействий.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Максимальное число кликов для метрики NoC.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Минимальное число кликов при оценке.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='Порог преобразования вероятностей в маску сегментации.')
    parser.add_argument('--clicks-limit', type=int, default=None, help='Лимит кликов на входе модели; -1 — значение --n-clicks')
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help='Режимы: cvpr, fixed<число> (например, fixed400, fixed600).')

    parser.add_argument('--save-ious', action='store_true', default=False, help='Сохранять значения IoU')
    parser.add_argument('--print-ious', action='store_true', default=False, help='Выводить средний IoU после каждого клика')
    parser.add_argument('--vis-preds', action='store_true', default=False, help='Сохранять визуализации предсказаний')
    group_checkpoints.add_argument('--vis_path', type=str, default='./experiments/vis_val/',
                                   help='Путь сохранения результатов оценки')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Имя модели для графиков.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='Путь к файлу конфигурации.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='Папка журналов оценки. По умолчанию: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--infer-size', type=int, default=384,
                        help='Размер входного изображения модели при предсказании')

    parser.add_argument('--target-crop-r', type=float, default=1.40,
                                  help='Коэффициент расширения целевой области')

    parser.add_argument('--focus-crop-r', type=float, default=1.40,
                                  help='Коэффициент расширения области уточнения')
    parser.add_argument('--optim_min', action='store_true', default=False, help='Минимизировать IoU при оптимизации')
    parser.add_argument('--vis_optim', action='store_true', default=False, help='Сохранять визуализации шагов оптимизации')
    parser.add_argument('--n_opt_steps', type=int, default=10, help='Число шагов оптимизации положения клика')
    parser.add_argument('--lr_mult', type=float, default=1, help='Множитель скорости обучения при оптимизации')
    parser.add_argument('--n_workers', type=int, default=1, help='Число параллельных процессов оценки')
    parser.add_argument('--deterministic', action='store_true', default=False,
                            help='Включить детерминированную оценку')
    parser.add_argument('--n_samples', type=int, default=0, help='Использовать первые N примеров; 0 — все примеры')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg




def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  infer_size = 256,
                  focus_crop_r= 1.4,
                  with_flip=False,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if predictor_params is not None:
        predictor_params_.update(predictor_params)

    predictor = BaselinePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, infer_size =infer_size, **predictor_params_)

    return predictor



def main():
    args, cfg = parse_args()
    if args.deterministic:
        setup_deterministic(seed=0)

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    print('Файлы весов: ', checkpoints_list)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Анализ IoU доступен только для одного файла весов"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg, args)

        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device)

            predictor_params, zoomin_params, infer_size = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,
                                      infer_size=infer_size,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      focus_crop_r = args.focus_crop_r,
                                      zoom_in_params=zoomin_params)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback, args=args, support_deterministic=args.deterministic)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)

            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)
            print_header = False


def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    if args.eval_mode == 'cvpr':
        zoom_in_params = {

            'skip_clicks': 0,
            'target_size': 600 if dataset_name == 'DAVIS' else 400,
            'expansion_ratio': args.target_crop_r
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    infer_size = 384

    return predictor_params, zoom_in_params,infer_size


if __name__ == '__main__':
    main()
