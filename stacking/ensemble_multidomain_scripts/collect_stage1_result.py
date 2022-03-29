
import os
from pathlib import Path
import itertools

import pandas as pd
import itertools
import yaml

K_CV = 5

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datatrack')
    parser.add_argument('feat_type', default='weak36')
    return parser.parse_args()


def get_learner_data(stage1_result_dir, pred_datatrack, use_cv_result, use_upper_lower, column_tag, k_cv=K_CV):

    df_vals = []
    df_tests = []
    for i_cv in range(k_cv):
        if use_cv_result:
            df_vals.append(pd.read_csv(stage1_result_dir / str(i_cv) / f'val.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(stage1_result_dir / str(i_cv) / f'test.csv',
                            index_col=0))
        else:
            pred_dir = stage1_result_dir / str(i_cv) / f'pred-{pred_datatrack}'
            df_vals.append(pd.read_csv(pred_dir / f'train.csv',
                            index_col=0))
            df_tests.append(pd.read_csv(pred_dir / f'test.csv',
                            index_col=0))

    if use_cv_result:
        df_train = pd.concat(df_vals)
    else:
        df_train = sum(df_vals) / len(df_vals)
    df_test = sum(df_tests) / len(df_tests)

    # empty column df
    df_train_new = df_train[[]].copy()
    df_test_new = df_test[[]].copy()

    if use_upper_lower:
        col_name = 'mean-' + column_tag
        df_train_new[col_name] = df_train['pred_mos'].copy()
        df_test_new[col_name] = df_test['pred_mos'].copy()

        col_name = 'lower-' + column_tag
        df_train_new[col_name] = df_train['lower_mos'].copy()
        df_test_new[col_name] = df_test['lower_mos'].copy()

        col_name = 'upper-' + column_tag
        df_train_new[col_name] = df_train['upper_mos'].copy()
        df_test_new[col_name] = df_test['upper_mos'].copy()

    else:
        col_name = 'pred-' + column_tag
        df_train_new[col_name] = df_train['pred_mos'].copy()
        df_test_new[col_name] = df_test['pred_mos'].copy()

    return df_train_new, df_test_new


def main():

    args = get_arg()

    stage2_data_dir = Path('../out/ensemble-multidomain/data-stage2') / args.datatrack / args.feat_type
    stage1_result_base_dir = Path('../out/ensemble-multidomain/stage1')

    feat_conf = yaml.safe_load(open('./stage2-method/{}.yaml'.format(args.feat_type)))
    print(feat_conf)

    df_train_list = []
    df_test_list = []

    for strong_learner in feat_conf['strong_learners']:

        stage1_result_dir = stage1_result_base_dir / args.datatrack / strong_learner

        column_tag = strong_learner
        k_cv = 3 if args.datatrack == 'phase1-ood' else K_CV
        df_train, df_test = get_learner_data(stage1_result_dir=stage1_result_dir,
                                                pred_datatrack=args.datatrack,
                                                use_cv_result=True,
                                                use_upper_lower=False,
                                                column_tag=strong_learner,
                                                k_cv=k_cv)

        df_train_list.append(df_train)
        df_test_list.append(df_test)

    for train_datatrack, model_type, ssl_type in itertools.product(
                            feat_conf['weak_learners']['datatracks'],
                            feat_conf['weak_learners']['model_types'],
                            feat_conf['weak_learners']['ssl_types']):

        if model_type == 'autogp':
            if train_datatrack == 'phase1-main':
                model_type = 'svgp'
            else:
                model_type = 'exactgp'

        use_cv_result = (args.datatrack == train_datatrack or train_datatrack.startswith('phase1-all'))
        use_upper_lower = (model_type in ['svgp', 'exactgp'])

        stage1_result_dir = stage1_result_base_dir / train_datatrack / f'{model_type}-{ssl_type}'

        column_tag = f'{train_datatrack}---{model_type}---{ssl_type}'

        df_train, df_test = get_learner_data(stage1_result_dir=stage1_result_dir,
                                                pred_datatrack=args.datatrack,
                                                use_cv_result=use_cv_result,
                                                use_upper_lower=use_upper_lower,
                                                column_tag=column_tag)

        df_train_list.append(df_train)
        df_test_list.append(df_test)

    df_train_all = pd.concat(df_train_list, axis=1)
    df_test_all = pd.concat(df_test_list, axis=1)

    df_train_all.sort_index(inplace=True)
    df_test_all.sort_index(inplace=True)


    print('Columns: {}'.format(df_train_all.columns))
    print('Train: {}'.format(df_train_all.shape))
    print('Test: {}'.format(df_test_all.shape))

    os.makedirs(stage2_data_dir, exist_ok=True)
    df_train_all.to_csv(stage2_data_dir / 'train-X.csv')
    df_test_all.to_csv(stage2_data_dir / 'test-X.csv')


if __name__ == '__main__':
    main()


