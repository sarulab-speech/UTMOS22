
import os
from pathlib import Path
import logging
from logging import getLogger
import random
import json

import numpy as np
import torch

from data_util import load_stage1_data

from models import Ridge, LinearSVR, KernelSVR, LightGBM, RandomForest
from gp_models import SVGP, ExactGP

logger = getLogger(__name__)

RAND_SEED = 0

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('datatrack')
    parser.add_argument('ssl_type')
    parser.add_argument('i_cv', type=int)
    return parser.parse_args()




def main():
    args = get_arg()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    data = load_stage1_data(
            datatrack=args.datatrack,
            ssl_type=args.ssl_type,
            i_cv=args.i_cv,
        )

    if args.method == 'svgp':
        model = SVGP()
    elif args.method == 'exactgp':
        model = ExactGP()
    elif args.method == 'rf':
        model = RandomForest()
    else:
        param_file = Path('../out/ensemble-multidomain/opt_hp_stage1') / args.datatrack / \
                        f'{args.method}-{args.ssl_type}' / 'params.json'
        params = json.load(open(param_file, 'rb'))
        logger.info('Params: {}'.format(params))

        if args.method == 'ridge':
            model = Ridge(params=params)
        elif args.method == 'linear_svr':
            model = LinearSVR(params=params)
        elif args.method == 'kernel_svr':
            model = KernelSVR(params=params)
        elif args.method == 'lightgbm':
            model = LightGBM(params=params)
        else:
            raise RuntimeError('Not supported method: "{}"'.format(args.method))

    model.train(data['train']['X'], data['train']['y'],
                                    data['val']['X'], data['val']['y'])

    df_val = model.predict(data['val']['X'], data['val']['df'])
    df_test = model.predict(data['test']['X'], data['test']['df'])

    out_dir = Path('../out/ensemble-multidomain/stage1') / args.datatrack / \
                    f'{args.method}-{args.ssl_type}' / str(args.i_cv)
    os.makedirs(out_dir, exist_ok=True)

    df_val.to_csv(out_dir / 'val.csv')
    df_test.to_csv(out_dir / 'test.csv')

    model.save_model(out_dir)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{name}: {message}', style='{')
    main()


