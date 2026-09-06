# Исследование устойчивости моделей интерактивной сегментации к состязательным кликовым подсказкам

Код для оценки устойчивости моделей интерактивной сегментации к положению кликовых подсказок.

<img width="1772" height="289" alt="image" src="https://github.com/user-attachments/assets/9ab24f1f-6956-4c04-927e-85f0322b9714" />

## Подготовка окружения

Выполняйте команды из корня репозитория. Исходное окружение использует PyTorch 1.13.1 и CUDA 11.7. Команды ниже рассчитаны на Bash и установленную Conda.

```bash
pip3 install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
conda install -y scikit-image
conda install -y -c anaconda cmake
pip install -r requirements.txt
pip install svgwrite svgpathtools cssutils numba torch-tools visdom einops tensorboard easydict segmentation_models_pytorch opt_einsum joblib albumentations==1.3.1 numpy==1.23.5

git clone https://github.com/BachiLi/diffvg.git --recursive
cd diffvg
python setup.py install
cd ..
```

## Датасеты и веса моделей

Проект использует структуру датасетов и сценарии оценки на основе [RITM](https://github.com/supervisely-ecosystem/ritm-interactive-segmentation). Скачайте и распакуйте датасеты, затем укажите пути к ним в [config.yml](config.yml).

| Датасет | Состав тестовой выборки | Скачать |
|---|---|---|
| GrabCut | 50 изображений, по одному объекту на изображение | [GrabCut.zip, 11 МБ][GrabCut] |
| Berkeley | 96 изображений, 100 объектов | [Berkeley.zip, 7 МБ][Berkeley] |
| DAVIS | 345 изображений, по одному объекту на изображение | [DAVIS.zip, 43 МБ][DAVIS] |
| COCO-MVal | 800 изображений, 800 объектов | [COCO_MVal.zip, 127 МБ][COCO_MVal] |

[GrabCut]: https://drive.google.com/file/d/1tU17eJaevYd5PQAwzEK6oG1N9eBJORC4/view?usp=sharing
[Berkeley]: https://drive.google.com/file/d/1yStolKW8AnS5rYm3VhgnID_XReNCXbEJ/view?usp=sharing
[DAVIS]: https://drive.google.com/file/d/1DhwMOqbwH4tcIrVfqtSczdofjdHHnMCJ/view?usp=sharing
[COCO_MVal]: https://drive.google.com/file/d/1oz1VT8YqykQFfx_Vtp_wtYFGz5mYBOnp/view?usp=sharing

```yaml
INTERACTIVE_MODELS_PATH: ""
EXPS_PATH: "./experiments"
GRABCUT_PATH: "../datasets/GrabCut"
BERKELEY_PATH: "../datasets/Berkeley"
DAVIS_PATH: "../datasets/DAVIS"
COCO_MVAL_PATH: "../datasets/COCO_MVal"
```

Пути отсчитываются от текущей рабочей папки. Для DAVIS и COCO-MVal загрузчик ожидает подпапки `img` и `gt`. В командах используйте `COCO-MVal`; прежнее написание `COCO_MVal` также поддерживается для совместимости с существующими результатами.

Скачайте веса из репозиториев соответствующих моделей. Для запуска [run.sh](run.sh) разместите веса в папке `MODEL_CHECKPOINTS` с путями, указанными в сценарии. Имена файлов весов сохранены для совместимости с исходными моделями.

## Запуск оптимизации

Пример оценки RITM с минимизацией IoU. Замените путь после `--checkpoint` на путь к вашему файлу весов:

```bash
python3 scripts/evaluate_model_ritm.py NoBRS \
  --checkpoint=./MODEL_CHECKPOINTS/RITM/coco_lvis_h18_itermask.pth \
  --datasets=GrabCut,Berkeley,DAVIS,COCO-MVal \
  --deterministic --print-ious --iou-analysis --save-ious \
  --thresh=0.49 --n-clicks=10 --n_opt_steps=11 \
  --n_workers=1 --lr_mult=1 --optim_min
```

| Параметр | Назначение |
|---|---|
| `--datasets` | Датасеты через запятую; по умолчанию используются все четыре |
| `--n_opt_steps` | Число шагов оптимизации положения каждого клика |
| `--optim_min` | Минимизация IoU; без флага выполняется максимизация |
| `--lr_mult` | Множитель скорости обучения; для базовой стратегии задайте `0` вместе с `--n_opt_steps=1` |
| `--n_workers` | Число параллельных процессов оценки; выбирайте с учётом памяти видеокарты |
| `--deterministic` | Включение детерминированных алгоритмов PyTorch; некоторые модели используют недетерминированные операции |
| `--n-clicks` | Максимальное число кликов |
| `--n_samples` | Ограничение числа примеров для отладки; `0` означает всю выборку |
| `--iou-analysis` | Сохранение траекторий IoU и BIoU для анализа в ноутбуке |
| `--config-path` | Путь к конфигурации; по умолчанию `./config.yml` |

Остальные параметры доступны через `--help` у соответствующего сценария оценки.

### Оценка всех моделей

Для [SAM](https://github.com/facebookresearch/segment-anything), [SAM-HQ](https://github.com/SysCV/sam-hq) и [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) нужны отдельные пакеты с поддержкой обратного распространения градиентов. Удалите блокирующие его вызовы `torch.no_grad` в используемых путях вычислений либо скачайте [подготовленные версии пакетов](https://drive.google.com/file/d/1DsBMiTJENqwlQSny-83oWvMBy4QJS59S/view?usp=sharing).

После распаковки архива выполните из папки с тремя пакетами:

```bash
pip install -e ./segment-anything-custom-build
pip install -e ./sam-hq-custom-build
pip install -e ./MobileSAM-custom-build
```

Сценарий [run.sh](run.sh) содержит команды для SAM, SAM-HQ, MobileSAM, [RITM](https://github.com/supervisely-ecosystem/ritm-interactive-segmentation), [SimpleClick](https://github.com/uncbiag/SimpleClick/), [GPCIS](https://github.com/zmhhmz/GPCIS_CVPR2023), [CDNet](https://github.com/XavierCHEN34/ClickSEG) и [CFR-ICL](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/). После подготовки окружения и весов вернитесь в корень репозитория и запустите:

```bash
bash run.sh
```

Сценарий выполняет минимизацию и максимизацию на всех четырёх датасетах. Для базовых результатов повторите нужные команды с `--lr_mult=0 --n_opt_steps=1`. Папки результатов базовых запусков переименуйте, убрав конечный суффикс `_MIN` или `_MAX`, прежде чем запускать оптимизацию с теми же весами: иначе файлы будут перезаписаны.

## Расчёт метрик

По умолчанию результаты записываются в `./experiments/evaluation_logs/others`, то есть в подпапку `EXPS_PATH` из конфигурации. Откройте [Evaluate Models.ipynb](Evaluate%20Models.ipynb), проверьте `exp_path`, выберите модели в `models_to_print` и выполните ячейки по порядку.

Ноутбук рассчитывает IoU и BIoU для минимизации, максимизации и базовой стратегии, а также разницу между максимальным и минимальным значениями. Если базовый запуск отсутствует, для него выводится «нет данных». Старые результаты с именем `COCO_MVal` читаются как COCO-MVal; результаты других датасетов пропускаются.

Папка, указанная в `exp_path`, должна содержать каталоги моделей следующего вида:

```text
имя_модели/plots/        # Базовая стратегия, если рассчитана
имя_модели_MIN/plots/    # Минимизация IoU
имя_модели_MAX/plots/    # Максимизация IoU
```
