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
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.inference.utils import setup_deterministic

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
                        help='Режимы: cvpr, fixed<число> (например, fixed400, fixed600).')
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
    parser.add_argument('--n_opt_steps', type=int, default=11, help='Число шагов оптимизации положения клика')
    parser.add_argument('--lr_mult', type=float, default=1, help='Множитель скорости обучения при оптимизации')
    parser.add_argument('--n_workers', type=int, default=1, help='Число параллельных процессов оценки')
    parser.add_argument('--n_samples', type=int, default=0, help='Использовать первые N примеров для отладки; 0 — все примеры')

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

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
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
            'target_size': 600 if dataset_name == 'DAVIS' else 400
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Некорректный путь к эксперименту."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pt*'), reverse=True)
        assert len(checkpoints_list) > 0, "Не найдены файлы весов."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, Path(str(logs_path) + "_" + ("MIN" if args.optim_min else "MAX")), logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    all_ious = np.array(all_ious, dtype=object)[:, 1]

    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)[:, 0]
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                             for click_id in list(range(1, 100)) if click_id <= min_num_clicks])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious
        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback


if __name__ == '__main__':
    main()
