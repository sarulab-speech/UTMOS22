
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

def evaluate_val(df_val):

    assert len(df_val) == len(df_val.index.unique())

    mse = np.mean(np.square(df_val['true_mos'] - df_val['pred_mos']))
    utt_srcc = scipy.stats.spearmanr(df_val['true_mos'], df_val['pred_mos'])[0]
    print('CV UTT MSE: {:f}'.format(mse))
    print('CV UTT SRCC: {:f}'.format(utt_srcc))

    # sys
    df_val['system_ID'] = df_val.index.str.extract(r'^(.+?)-').values

    df_val_sys = df_val.groupby('system_ID')['pred_mos'].mean()
    df_true_sys = df_val.groupby('system_ID')['true_mos'].mean()

    df_sys = pd.merge(df_val_sys, df_true_sys, on='system_ID', how='left')

    sys_mse = np.mean(np.square(df_sys['true_mos'] - df_sys['pred_mos']))
    sys_srcc = scipy.stats.spearmanr(df_sys['true_mos'], df_sys['pred_mos'])[0]
    print('CV SYS MSE:  {:f}'.format(sys_mse))
    print('CV SYS SRCC: {:f}'.format(sys_srcc))


    return {'cv_utt_mse': mse}

def evaluate_test(df_test, pred_datatrack):

    utt_mse = np.mean(np.square(df_test['true_mos'] - df_test['pred_mos']))
    utt_srcc = scipy.stats.spearmanr(df_test['true_mos'], df_test['pred_mos'])[0]
    print('TEST UTT MSE:  {:f}'.format(utt_mse))
    print('TEST UTT SRCC: {:f}'.format(utt_srcc))

    df_test['system_ID'] = df_test.index.str.extract(r'^(.+?)-').values

    df_test_sys = df_test.groupby('system_ID')['pred_mos'].mean()

    df_true_sys = pd.read_csv(f'../data/{pred_datatrack}/DATA/mydata_system.csv')

    df_sys = pd.merge(df_test_sys, df_true_sys, on='system_ID', how='left').set_index('system_ID')

    sys_mse = np.mean(np.square(df_sys['mean'] - df_sys['pred_mos']))
    sys_srcc = scipy.stats.spearmanr(df_sys['mean'], df_sys['pred_mos'])[0]
    print('TEST SYS MSE:  {:f}'.format(sys_mse))
    print('TEST SYS SRCC: {:f}'.format(sys_srcc))

    return {'test_utt_mse': utt_mse,
            'test_utt_srcc': utt_srcc,
            'test_sys_mse': sys_mse,
            'test_sys_srcc': sys_srcc}



def calc_strong_learner_score(pred_datatrack, model_type, k_cv=K_CV):

    print('Stage1, {}, {}'.format(pred_datatrack, model_type))

    result_dir = Path('../out/ensemble-multidomain/stage1') / \
                            pred_datatrack / f'{model_type}'
    print(result_dir)

    use_cv_result = True

    df_vals = []
    df_tests = [] 
    
    for i_cv in range(k_cv):
        if use_cv_result:
            df_vals.append(pd.read_csv(result_dir / str(i_cv) / f'val.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(result_dir / str(i_cv) / f'test.csv',
                            index_col=0))
        else:
            pred_dir = result_dir / str(i_cv) / f'pred-{pred_datatrack}'
            df_vals.append(pd.read_csv(pred_dir / f'train.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(pred_dir / f'test.csv',
                            index_col=0))

    if use_cv_result:
        df_val = pd.concat(df_vals)
    else:
        df_val = sum(df_vals) / len(df_vals)
    df_test = sum(df_tests) / len(df_tests)

    result = {'stage': 'stage1', 'train_datatrack': 'all',
                'model_type': model_type, 'feat_type': 'nn'}

    result.update(evaluate_val(df_val))
    result.update(evaluate_test(df_test, pred_datatrack))

    return result

def calc_stage1_score(pred_datatrack, train_datatrack, model_type, ssl_type):

    print('Stage1, {}, {}, {}, {}'.format(pred_datatrack, train_datatrack, model_type, ssl_type))

    result_dir = Path('../out/ensemble-multidomain/stage1') / \
                            train_datatrack / f'{model_type}-{ssl_type}'
    print(result_dir)

    use_cv_result = (pred_datatrack == train_datatrack or train_datatrack.startswith('phase1-all'))

    df_vals = []
    df_tests = [] 
    
    for i_cv in range(K_CV):
        if use_cv_result:
            df_vals.append(pd.read_csv(result_dir / str(i_cv) / f'val.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(result_dir / str(i_cv) / f'test.csv',
                            index_col=0))
        else:
            pred_dir = result_dir / str(i_cv) / f'pred-{pred_datatrack}'
            df_vals.append(pd.read_csv(pred_dir / f'train.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(pred_dir / f'test.csv',
                            index_col=0))

    if use_cv_result:
        df_val = pd.concat(df_vals)
    else:
        df_val = sum(df_vals) / len(df_vals)
    df_test = sum(df_tests) / len(df_tests)

    result = {'stage': 'stage1', 'train_datatrack': train_datatrack,
                'model_type': model_type, 'feat_type': ssl_type}

    result.update(evaluate_val(df_val))
    result.update(evaluate_test(df_test, pred_datatrack))

    return result

def calc_stage_n_score(pred_datatrack, stage, model_type, feat_type):

    result_dir = Path('../out/ensemble-multidomain') / stage /  \
                            pred_datatrack / f'{model_type}-{feat_type}'
    print(result_dir)

    df_vals = []
    df_tests = [] 

    for i_cv in range(K_CV):
        df_vals.append(pd.read_csv(result_dir / str(i_cv) / f'val.csv',
                        index_col=0))
        df_tests.append(pd.read_csv(result_dir / str(i_cv) / f'test.csv',
                        index_col=0))

    df_val = pd.concat(df_vals)
    df_test = sum(df_tests) / len(df_tests)

    result = {'stage': stage, 'train_datatrack': pred_datatrack,
                'model_type': model_type, 'feat_type': feat_type}

    result.update(evaluate_val(df_val))
    result.update(evaluate_test(df_test, pred_datatrack))

    return result



def main():

    args = get_arg()

    feat_conf = yaml.safe_load(open('./stage2-method/{}.yaml'.format(args.feat_type)))
    print(feat_conf)

    data = []

    # stage1
    for model_type in feat_conf['strong_learners']:
        k_cv = 3 if args.datatrack == 'phase1-ood' else K_CV
        data.append(calc_strong_learner_score(args.datatrack, model_type, k_cv=k_cv))


    for train_datatrack, model_type, ssl_type in itertools.product(
                            feat_conf['weak_learners']['datatracks'],
                            feat_conf['weak_learners']['model_types'],
                            feat_conf['weak_learners']['ssl_types']):
        data.append(calc_stage1_score(args.datatrack, train_datatrack, model_type, ssl_type))

    # stage2
    for model_type in feat_conf['weak_learners']['model_types']:
        data.append(calc_stage_n_score(args.datatrack, 'stage2', model_type, args.feat_type))

    # stage3
    for model_type in ['ridge']:
        data.append(calc_stage_n_score(args.datatrack, 'stage3', model_type, args.feat_type))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'../out/ensemble-multidomain/result/{args.datatrack}-{args.feat_type}.csv')



if __name__ == '__main__':
    main()


