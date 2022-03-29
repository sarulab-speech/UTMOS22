
import os
from pathlib import Path
import logging
from logging import getLogger
import random
import json

import numpy as np
import torch

from data_util import load_stage2_data

from models import Ridge, LinearSVR, KernelSVR, LightGBM, RandomForest
from gp_models import SVGP, ExactGP

logger = getLogger(__name__)

RAND_SEED = 0

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('datatrack')
    parser.add_argument('feat_type')
    parser.add_argument('i_cv', type=int)
    parser.add_argument('--use_opt', action='store_true', default=False)
    return parser.parse_args()




def main():
    args = get_arg()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    data = load_stage2_data(
            datatrack=args.datatrack,
            feat_type=args.feat_type,
            i_cv=args.i_cv,
        )

    method = args.method

    if method == 'autogp':
        if args.datatrack == 'phase1-main':
            method = 'svgp'
        else:
            method = 'exactgp'

    if method == 'svgp':
        model = SVGP(stage='stage2')
    elif method == 'exactgp':
        model = ExactGP(stage='stage2')
    elif method == 'rf':
        model = RandomForest()
    else:
        if args.use_opt:
            param_file = Path('../out/ensemble-multidomain/opt_hp_stage2') / args.datatrack / \
                            f'{method}-{args.feat_type}' / 'params.json'
            params = json.load(open(param_file, 'rb'))
            logger.info('Params: {}'.format(params))
        else:
            params = {}

        if method == 'ridge':
            model = Ridge(params=params)
        elif method == 'linear_svr':
            model = LinearSVR(params=params)
        elif method == 'kernel_svr':
            model = KernelSVR(params=params)
        elif method == 'lightgbm':
            model = LightGBM(params=params)
        else:
            raise RuntimeError('Not supported method: "{}"'.format(method))

    model.train(data['train']['X'], data['train']['y'],
                                    data['val']['X'], data['val']['y'])

    df_val = model.predict(data['val']['X'], data['val']['df'])
    df_test = model.predict(data['test']['X'], data['test']['df'])

    out_dir = Path('../out/ensemble-multidomain/stage2') / args.datatrack / \
                    f'{method}-{args.feat_type}' / str(args.i_cv)
    os.makedirs(out_dir, exist_ok=True)

    df_val.to_csv(out_dir / 'val.csv')
    df_test.to_csv(out_dir / 'test.csv')

    model.save_model(out_dir)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{name}: {message}', style='{')
    main()


