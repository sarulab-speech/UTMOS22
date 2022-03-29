
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


    df_test['pred_mos'].to_csv(f'../out/ensemble-multidomain/answer/{args.datatrack}-{args.feat_type}.csv',
                header=None)

    #
    if args.datatrack == 'testphase-ood':
        post_testphase_path = '../post-testphase-distribution/ood/test_mos_list.txt'
    elif  args.datatrack == 'testphase-main':
        post_testphase_path = '../post-testphase-distribution/main/test_mos_list.txt'
    else:
        raise NotImplementedError()
    true_mos = pd.read_csv(post_testphase_path, header=None,
                            names=['wavname', 'true_mos'], index_col=0)

    df_test['true_mos'] = true_mos['true_mos']

    mse = np.mean(np.square(df_test['true_mos'] - df_test['pred_mos']))
    utt_srcc = scipy.stats.spearmanr(df_test['true_mos'], df_test['pred_mos'])[0]
    print('CV UTT MSE: {:f}'.format(mse))
    print('CV UTT SRCC: {:f}'.format(utt_srcc))

    # sys
    df_test['system_ID'] = df_test.index.str.extract(r'^(.+?)-').values

    df_val_sys = df_test.groupby('system_ID')['pred_mos'].mean()
    df_true_sys = df_test.groupby('system_ID')['true_mos'].mean()

    df_sys = pd.merge(df_val_sys, df_true_sys, on='system_ID', how='left')

    sys_mse = np.mean(np.square(df_sys['true_mos'] - df_sys['pred_mos']))
    sys_srcc = scipy.stats.spearmanr(df_sys['true_mos'], df_sys['pred_mos'])[0]
    print('CV SYS MSE:  {:f}'.format(sys_mse))
    print('CV SYS SRCC: {:f}'.format(sys_srcc))



if __name__ == '__main__':
    main()


