
import os
from pathlib import Path
import logging
from logging import getLogger
import random
import json

import numpy as np
import torch

from data_util import load_stage1_data, load_stage1_train_all_data, load_stage1_test_data

from models import Ridge, LinearSVR, KernelSVR, LightGBM, RandomForest
from gp_models import SVGP, ExactGP

logger = getLogger(__name__)

RAND_SEED = 0

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('train_datatrack')
    parser.add_argument('ssl_type')
    parser.add_argument('i_cv', type=int)
    parser.add_argument('pred_datatrack')
    return parser.parse_args()


def main():
    args = get_arg()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    if args.method == 'svgp':
        model = SVGP()
    elif args.method == 'exactgp':
        model = ExactGP()
    elif args.method == 'rf':
        model = RandomForest()
    else:
        if args.method == 'ridge':
            model = Ridge()
        elif args.method == 'linear_svr':
            model = LinearSVR()
        elif args.method == 'kernel_svr':
            model = KernelSVR()
        elif args.method == 'lightgbm':
            model = LightGBM()
        else:
            raise RuntimeError('Not supported method: "{}"'.format(args.method))

    model_dir = Path('../out/ensemble-multidomain/stage1') / args.train_datatrack / \
                    f'{args.method}-{args.ssl_type}' / str(args.i_cv)
    out_dir = model_dir / f'pred-{args.pred_datatrack}'
    os.makedirs(out_dir, exist_ok=True)

    logger.info('Outdir: {}'.format(out_dir))

    if args.method == 'svgp':
        # train_data = load_stage1_data(
        #         datatrack=args.train_datatrack,
        #         ssl_type=args.ssl_type,
        #         i_cv=args.i_cv,
        #     )
        # model.load_model(model_dir, train_data['train']['X'])
        model.load_model(model_dir)
    elif args.method == 'exactgp':
        train_data = load_stage1_data(
                datatrack=args.train_datatrack,
                ssl_type=args.ssl_type,
                i_cv=args.i_cv,
            )
        model.load_model(model_dir, train_data['train']['X'],  train_data['train']['y'])
    else:
        model.load_model(model_dir)

    pred_data = {}
    pred_data['train'] = load_stage1_train_all_data(
            datatrack=args.pred_datatrack,
            ssl_type=args.ssl_type,
        )
    pred_data['test'] = load_stage1_test_data(
            datatrack=args.pred_datatrack,
            ssl_type=args.ssl_type,
        )

    df_train = model.predict(pred_data['train']['X'], pred_data['train']['df'])
    df_test = model.predict(pred_data['test']['X'], pred_data['test']['df'])

    df_train.to_csv(out_dir / 'train.csv')
    df_test.to_csv(out_dir / 'test.csv')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{name}: {message}', style='{')
    main()


