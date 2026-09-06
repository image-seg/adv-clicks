import sys
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from os.path import dirname, join

sys.path.insert(0, '.')

module_path = dirname(__file__)
sys.path.append(join(module_path, '../'))

from isegm.inference import utils
from isegm.inference.datasets import parse_dataset_names
from isegm.utils.exp import load_config_file
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from isegm.inference.utils import setup_deterministic
from evaluate_model_ritm import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback
from isegm.inference.transforms import ZoomIn
from isegm.inference.predictors.cfr import CFRPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Оценка устойчивости интерактивной сегментации к положению кликов.')

    parser.add_argument('--deterministic', action='store_true', default=False, help='Включить детерминированную оценку')

    parser.add_argument('mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='Режим предсказания маски')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='Путь к файлу весов. '
                                        'Допускается относительный путь от cfg.INTERACTIVE_MODELS_PATH '
                                        'или абсолютный путь. Расширение файла можно опустить.')
    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='Относительный путь к эксперименту с весами. '
                                        '(относительно cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=parse_dataset_names, default='GrabCut,Berkeley,DAVIS,COCO-MVal',
                        help='Датасеты для оценки модели. '
                             'Укажите имена через запятую: GrabCut, Berkeley, DAVIS, COCO-MVal')

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
                        help="Режимы: cvpr, fixed<число> или fixed<число>,<число> (например, fixed400, fixed400,600).")

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--save-ious', action='store_true', default=False, help='Сохранять значения IoU')
    parser.add_argument('--print-ious', action='store_true', default=False, help='Выводить средний IoU после каждого клика')
    parser.add_argument('--vis-preds', action='store_true', default=False, help='Сохранять визуализации предсказаний')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Имя модели для графиков.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='Путь к файлу конфигурации.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='Папка журналов оценки. По умолчанию: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--optim_min', action='store_true', default=False, help='Минимизировать IoU при оптимизации')
    parser.add_argument('--vis_optim', action='store_true', default=False, help='Сохранять визуализации шагов оптимизации')
    parser.add_argument('--n_opt_steps', type=int, default=10, help='Число шагов оптимизации положения клика')
    parser.add_argument('--lr_mult', type=float, default=1, help='Множитель скорости обучения при оптимизации')
    parser.add_argument('--n_workers', type=int, default=1, help='Число параллельных процессов оценки')
    parser.add_argument('--n_samples', type=int, default=0, help='Использовать первые N примеров; 0 — все примеры')
    parser.add_argument('--cf-n', default=0, type=int,
                        help='Число шагов каскадного уточнения')
    parser.add_argument('--cf-click', default=1, type=int,
                        help='Число кликов до каскадного уточнения')
    parser.add_argument('--acf', action='store_true', default=False,
                        help='Включить адаптивное каскадное уточнение')
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


def main():
    args, cfg = parse_args()

    if args.deterministic:
        setup_deterministic(seed=0)

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Анализ IoU доступен только для одного файла весов"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg, args)

        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device).eval()

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name, eval_ritm=args.eval_ritm)

            # Для моделей SimpleClick обычно требуется интерполяция позиционных эмбеддингов
            if not args.eval_ritm:
                interpolate_pos_embed_inference(model.backbone, zoomin_params['target_size'], args.device)

            predictor_params_ = {
                'optimize_after_n_clicks': 1
            }

            if zoomin_params is not None:
                zoom_in = ZoomIn(**zoomin_params)
            else:
                zoom_in = None

            if predictor_params is not None:
                predictor_params_.update(predictor_params)

            predictor = CFRPredictor(model, args.device, zoom_in=zoom_in, with_flip=True, **predictor_params_)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback, args=args, support_deterministic=True)

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


def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):

    predictor_params = {
        'cascade_step': args.cf_n + 1,
        'cascade_adaptive': args.acf,
        'cascade_clicks': args.cf_click
    }


    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448)
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size_h, crop_size_w)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params

if __name__ == '__main__':
    main()
