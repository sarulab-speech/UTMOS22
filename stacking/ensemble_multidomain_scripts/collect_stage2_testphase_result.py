
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


def get_learner_data(stage2_result_dir, pred_datatrack, use_upper_lower, column_tag, k_cv=K_CV):

    df_tests = []
    for i_cv in range(k_cv):
        pred_dir = stage2_result_dir / str(i_cv) / f'pred-{pred_datatrack}'
        df_tests.append(pd.read_csv(pred_dir / f'test.csv',
                        index_col=0))

    df_test = sum(df_tests) / len(df_tests)

    # empty column df
    df_test_new = df_test[[]].copy()

    if use_upper_lower:
        col_name = 'mean-' + column_tag
        df_test_new[col_name] = df_test['pred_mos'].copy()

        col_name = 'lower-' + column_tag
        df_test_new[col_name] = df_test['lower_mos'].copy()

        col_name = 'upper-' + column_tag
        df_test_new[col_name] = df_test['upper_mos'].copy()

    else:
        col_name = 'pred-' + column_tag
        df_test_new[col_name] = df_test['pred_mos'].copy()

    return df_test_new


def main():

    args = get_arg()

    stage3_data_dir = Path('../out/ensemble-multidomain/data-stage3') / args.datatrack / args.feat_type
    stage2_result_base_dir = Path('../out/ensemble-multidomain/stage2')

    feat_conf = yaml.safe_load(open('./stage2-method/{}.yaml'.format(args.feat_type)))
    print(feat_conf)

    df_test_list = []


    for model_type in feat_conf['weak_learners']['model_types']:

        if model_type == 'autogp':
            if train_datatrack == 'phase1-main':
                model_type = 'svgp'
            else:
                model_type = 'exactgp'

        use_upper_lower = (model_type in ['svgp', 'exactgp'])

        train_datatrack = 'phase1-main' if args.datatrack in ['testphase-main', 'valphase-main'] else 'phase1-ood'

        stage2_result_dir = stage2_result_base_dir / train_datatrack / f'{model_type}-{args.feat_type}'

        column_tag = model_type

        df_test = get_learner_data(stage2_result_dir, args.datatrack,
                                                    use_upper_lower, column_tag)
        df_test_list.append(df_test)

    df_test_all = pd.concat(df_test_list, axis=1)

    df_test_all.sort_index(inplace=True)


    print('Columns: {}'.format(df_test_all.columns))
    print('Test: {}'.format(df_test_all.shape))

    os.makedirs(stage3_data_dir, exist_ok=True)
    df_test_all.to_csv(stage3_data_dir / 'test-X.csv')


if __name__ == '__main__':
    main()


