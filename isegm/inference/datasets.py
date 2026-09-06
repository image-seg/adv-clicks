"""Имена датасетов, доступных для оценки моделей и анализа результатов."""

from argparse import ArgumentTypeError

SUPPORTED_DATASETS = ('GrabCut', 'Berkeley', 'DAVIS', 'COCO-MVal')


def normalize_dataset_name(name):
    """Проверить имя и привести прежнее написание COCO_MVal к COCO-MVal."""
    name = name.strip()
    if name == 'COCO_MVal':
        name = 'COCO-MVal'
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f'Неизвестный датасет: {name!r}. '
            f'Доступны: {", ".join(SUPPORTED_DATASETS)}.'
        )
    return name


def parse_dataset_names(value):
    """Проверить список датасетов, разделённых запятыми, для командной строки."""
    try:
        return ','.join(normalize_dataset_name(name) for name in value.split(','))
    except ValueError as error:
        raise ArgumentTypeError(str(error)) from error
