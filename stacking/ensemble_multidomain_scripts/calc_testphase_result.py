
import os
from pathlib import Path

import itertools

import yaml
import numpy as np
import pandas as pd
import scipy
import scipy.stats

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datatrack')
    parser.add_argument('feat_type', default='weak36')
    return parser.parse_args()

K_CV = 5


def main():

    args = get_arg()

    feat_conf = yaml.safe_load(open('./stage2-method/{}.yaml'.format(args.feat_type)))
    print(feat_conf)

    train_datatrack = 'phase1-main' if args.datatrack in ['testphase-main', 'valphase-main'] else 'phase1-ood'

    result_dir = Path('../out/ensemble-multidomain') / 'stage3' /  \
                            train_datatrack / f'ridge-{args.feat_type}'

    df_tests = [] 

    for i_cv in range(K_CV):
        df_tests.append(pd.read_csv(result_dir / str(i_cv) / f'pred-{args.datatrack}/test.csv',
                        index_col=0))

    df_test = sum(df_tests) / len(df_tests)

    answer_dir = Path('../out/ensemble-multidomain/answer')
    os.makedirs(answer_dir, exist_ok=True)
    df_test['pred_mos'].to_csv(answer_dir / f'{args.datatrack}-{args.feat_type}.csv',
                                header=None)



if __name__ == '__main__':
    main()


