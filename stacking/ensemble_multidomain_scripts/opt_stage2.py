
import os
from pathlib import Path
import logging
from logging import getLogger
import random

import numpy as np
import torch

from data_util import load_stage2_train_all_data

from models import Ridge, LinearSVR, KernelSVR, RandomForest, LightGBM
from gp_models import SVGP
import json

logger = getLogger(__name__)

RAND_SEED = 0

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('datatrack')
    parser.add_argument('feat_type')
    return parser.parse_args()




def main():
    args = get_arg()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    data = load_stage2_train_all_data(
            datatrack=args.datatrack,
            feat_type=args.feat_type,
        )

    if args.method == 'ridge':
        model = Ridge()
    elif args.method == 'linear_svr':
        model = LinearSVR(stage='stage2')
    elif args.method == 'kernel_svr':
        model = KernelSVR(stage='stage2')
    elif args.method == 'rf':
        raise NotImplementedError()
        # model = RandomForest()
    elif args.method == 'lightgbm':
        model = LightGBM()
    elif args.method == 'svgp':
        raise NotImplementedError()
    else:
        raise RuntimeError('Not supported method: "{}"'.format(args.method))

    best_params = model.optimize_hp(data['X'], data['y'])

    logger.info(best_params)

    out_dir = Path('../out/ensemble-multidomain/opt_hp_stage2') / args.datatrack / \
                    f'{args.method}-{args.feat_type}'
    os.makedirs(out_dir, exist_ok=True)

    with open(out_dir / 'params.json', encoding="utf-8", mode="w") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{name}: {message}', style='{')
    main()


