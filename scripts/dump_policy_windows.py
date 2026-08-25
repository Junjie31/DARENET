"""
Dump per-window predictions for Policy experiments.
Uses the unified DareNet + DareNetTrainer (hard-router, ffn_branch).

Usage:
    python scripts/dump_policy_windows.py \
        --ffn_branch spectral \
        --test_dataset MMPD \
        --checkpoint pretrained/PreTrainedModels/MMPD_MMPD_DareNet_spectral_seed42_Epoch29.pth \
        --output dumps/complex_mmpd.npz \
        --data_path /path/to/test/data
"""
import os, sys, argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from dataset import data_loader as data_loader_module
from neural_methods.trainer.DareNetTrainer import DareNetTrainer

logging.basicConfig(level=logging.INFO)

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffn_branch', required=True, choices=['spectral', 'gated_tf'])
    parser.add_argument('--test_dataset', required=True, choices=['MMPD', 'PURE', 'UBFC'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--cached_path', default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    if args.cached_path is None:
        args.cached_path = args.data_path
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = args.test_dataset

    # Auto-compute file_list_path for preprocessed data
    exp_data_name = "_".join([
        dataset, "SizeW128", "SizeH128", "ClipLength160",
        "DataTypeStandardized", "DataAugNone", "LabelTypeStandardized",
        "Crop_faceTrue", "Large_boxTrue", "Large_size1.5",
        "Dyamic_DetFalse", "det_len30", "Median_face_boxFalse"
    ])
    file_list_path = os.path.join(args.cached_path, 'DataFileLists',
                                   exp_data_name + '_0.0_1.0.csv')

    # Config-like object — simplified, only what trainer needs
    def _obj(**kw): return type('_', (), kw)()
    config = _obj(
        DEVICE=device,
        NUM_OF_GPU_TRAIN=1,
        TOOLBOX_MODE='only_test',
        TRAIN=_obj(
            EPOCHS=30, BATCH_SIZE=args.batch_size, LR=3e-4,
            MODEL_FILE_NAME='policy_dump', AUG=0, PLOT_LOSSES_AND_LR=False,
            SEED=42,
            OPTIMIZER=_obj(EPS=1e-4, BETAS=(0.9,0.999), MOMENTUM=0.9),
            DATA=_obj(
                FS=30, DATASET=dataset,
                DATA_PATH=args.data_path, CACHED_PATH=args.cached_path,
                EXP_DATA_NAME='', BEGIN=0.0, END=1.0,
                DO_PREPROCESS=False, DATA_FORMAT='NDCHW',
                FILE_LIST_PATH=file_list_path,
                PREPROCESS=_obj(
                    DATA_TYPE=['Standardized'], LABEL_TYPE='Standardized',
                    DO_CHUNK=True, CHUNK_LENGTH=160,
                    CROP_FACE=_obj(
                        DO_CROP_FACE=True, USE_LARGE_FACE_BOX=True,
                        LARGE_BOX_COEF=1.5,
                        DETECTION=_obj(DO_DYNAMIC_DETECTION=False,
                                       DYNAMIC_DETECTION_FREQUENCY=30,
                                       USE_MEDIAN_FACE_BOX=False)),
                    RESIZE=_obj(W=128, H=128),
                    BIGSMALL=_obj(BIG_DATA_TYPE=[''], SMALL_DATA_TYPE=[''],
                                  RESIZE=_obj(BIG_W=144, BIG_H=144, SMALL_W=9, SMALL_H=9))),
                FOLD=_obj(FOLD_NAME='', FOLD_PATH=''),
                INFO=_obj(LIGHT=[1,2,3,4], MOTION=[1,2,3,4], EXERCISE=[1,2],
                          SKIN_COLOR=[3,4,5,6], GENDER=[1,2], GLASSER=[1,2],
                          HAIR_COVER=[1,2], MAKEUP=[1,2]),
                FILTERING=_obj(USE_EXCLUSION_LIST=False, EXCLUSION_LIST=[''],
                               SELECT_TASKS=False, TASK_LIST=['']))),
        VALID=_obj(DATA=_obj(FS=30, DATASET=None, DATA_PATH='', CACHED_PATH='',
                              BEGIN=0.0, END=1.0, FILE_LIST_PATH=file_list_path,
                              PREPROCESS=_obj(DATA_TYPE=[''], LABEL_TYPE='', DO_CHUNK=True,
                                              CHUNK_LENGTH=160,
                                              CROP_FACE=_obj(DO_CROP_FACE=True, USE_LARGE_FACE_BOX=True,
                                                             LARGE_BOX_COEF=1.5,
                                                             DETECTION=_obj(DO_DYNAMIC_DETECTION=False,
                                                                            DYNAMIC_DETECTION_FREQUENCY=30,
                                                                            USE_MEDIAN_FACE_BOX=False)),
                                              RESIZE=_obj(W=128, H=128)),
                              FOLD=_obj(FOLD_NAME='', FOLD_PATH=''),
                              INFO=_obj(LIGHT=[''], MOTION=[''], EXERCISE=[True],
                                        SKIN_COLOR=[1], GENDER=[''], GLASSER=[True],
                                        HAIR_COVER=[True], MAKEUP=[True]),
                              FILTERING=_obj(USE_EXCLUSION_LIST=False, EXCLUSION_LIST=[''],
                                             SELECT_TASKS=False, TASK_LIST=['']))),
        TEST=_obj(
            USE_LAST_EPOCH=True, OUTPUT_SAVE_DIR='',
            METRICS=['MAE','RMSE','MAPE','Pearson','SNR'],
            DATA=_obj(
                FS=30, DATASET=dataset,
                DATA_PATH=args.data_path, CACHED_PATH=args.cached_path,
                EXP_DATA_NAME='', BEGIN=0.0, END=1.0,
                DO_PREPROCESS=False, DATA_FORMAT='NDCHW',
                FILE_LIST_PATH=file_list_path,
                PREPROCESS=_obj(
                    DATA_TYPE=['Standardized'], LABEL_TYPE='Standardized',
                    DO_CHUNK=True, CHUNK_LENGTH=160,
                    CROP_FACE=_obj(DO_CROP_FACE=True, USE_LARGE_FACE_BOX=True,
                                   LARGE_BOX_COEF=1.5,
                                   DETECTION=_obj(DO_DYNAMIC_DETECTION=False,
                                                  DYNAMIC_DETECTION_FREQUENCY=30,
                                                  USE_MEDIAN_FACE_BOX=False)),
                    RESIZE=_obj(W=128, H=128)),
                FOLD=_obj(FOLD_NAME='', FOLD_PATH=''),
                INFO=_obj(LIGHT=[1,2,3,4], MOTION=[1,2,3,4], EXERCISE=[1,2],
                          SKIN_COLOR=[3,4,5,6], GENDER=[1,2], GLASSER=[1,2],
                          HAIR_COVER=[1,2], MAKEUP=[1,2]),
                FILTERING=_obj(USE_EXCLUSION_LIST=False, EXCLUSION_LIST=[''],
                               SELECT_TASKS=False, TASK_LIST=['']))),
        MODEL=_obj(NAME='DareNet', DROP_RATE=0.2, FFN_BRANCH=args.ffn_branch,
                   MODEL_DIR=os.path.dirname(args.checkpoint)),
        INFERENCE=_obj(BATCH_SIZE=args.batch_size, EVALUATION_METHOD='FFT',
                       EVALUATION_WINDOW=_obj(USE_SMALLER_WINDOW=False, WINDOW_SIZE=10),
                       MODEL_PATH=args.checkpoint),
        UNSUPERVISED=_obj(METHOD=[], METRICS=[],
                          DATA=_obj(FS=30, DATASET='', DATA_PATH='', CACHED_PATH='', BEGIN=0.0,
                                    END=1.0, FILE_LIST_PATH=file_list_path,
                                    PREPROCESS=_obj(DATA_TYPE=[''], LABEL_TYPE='', DO_CHUNK=True,
                                                    CHUNK_LENGTH=160,
                                                    CROP_FACE=_obj(DO_CROP_FACE=True, USE_LARGE_FACE_BOX=True,
                                                                   LARGE_BOX_COEF=1.5,
                                                                   DETECTION=_obj(DO_DYNAMIC_DETECTION=False,
                                                                                  DYNAMIC_DETECTION_FREQUENCY=30,
                                                                                  USE_MEDIAN_FACE_BOX=False)),
                                                    RESIZE=_obj(W=128, H=128)),
                                    FOLD=_obj(FOLD_NAME='', FOLD_PATH=''))),
        LOG=_obj(PATH='runs/exp'),
    )

    # Build test data loader
    loader_map = {
        'UBFC': data_loader_module.UBFCrPPGLoader.UBFCrPPGLoader,
        'PURE': data_loader_module.PURELoader.PURELoader,
        'MMPD': data_loader_module.MMPDLoader.MMPDLoader,
    }
    if dataset not in loader_map:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg_data = config.TEST.DATA
    test_data = loader_map[dataset](name="test", data_path=args.data_path, config_data=cfg_data)
    test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=args.batch_size,
                             shuffle=False, generator=general_generator)
    data_loader_dict = {'test': test_loader, 'train': None, 'valid': None}

    # Create trainer and dump
    print(f"[Policy Dump] ffn_branch={args.ffn_branch}, test={dataset}, "
          f"ckpt={os.path.basename(args.checkpoint)}")
    trainer = DareNetTrainer(config, data_loader_dict)
    trainer.test_and_dump_windows(data_loader_dict, args.output)


if __name__ == '__main__':
    main()
