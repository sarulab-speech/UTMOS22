import pandas as pd
import numpy as np
from pathlib import Path
import os


def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datatrack')
    parser.add_argument('learner')
    return parser.parse_args()


def get_merge_df(df_true, df_pred):

    df = pd.merge(df_pred, df_true, on="wavname", how="left")
    df = df.set_index('wavname')[['pred_mos', 'true_mos']]
    
    return df

def get_true_mos(datatrack):

    train_true_path = f'../data/{datatrack}/DATA/sets/train_mos_list.txt'
    val_true_path = f'../data/{datatrack}/DATA/sets/val_mos_list.txt'

    df_true_dict = {}
    df_true_dict['train'] = pd.read_csv(train_true_path, header=None,
                                names=['wavname', 'true_mos'])
    df_true_dict['val'] = pd.read_csv(val_true_path, header=None,
                                names=['wavname', 'true_mos'])
    df_true_dict['train'].shape, df_true_dict['val'].shape

    return df_true_dict


def main():

    args = get_arg()

    phase = 'main'
    in_dir = Path('../strong_learner_result') / args.learner
    out_base_dir = Path(f'../out/ensemble-multidomain/stage1') \
                    / args.datatrack / args.learner

    df_true_dict = get_true_mos(args.datatrack)

    k_cv = 3 if args.datatrack == 'phase1-ood' else 5
    answer_file_tag = 'answer-ood' if args.datatrack == 'phase1-ood' else 'answer-main'

    for i_cv in range(k_cv):
        out_dir = out_base_dir / str(i_cv)
        os.makedirs(out_dir, exist_ok=True)
        print(out_dir)

        for split in ['train', 'val']:
            stacking_split_name = {'train': 'val', 'val': 'test'}[split]
            learner_split_name = {'train': 'fold', 'val': 'val'}[split]
            in_path = in_dir / f'{answer_file_tag}.csv{learner_split_name}_{i_cv}'

            df_pred = pd.read_csv(in_path, header=None,
                        names=['wavbase', 'pred_mos'])
            df_pred["wavname"] = df_pred["wavbase"] + ".wav"

            df = get_merge_df(df_true_dict[split], df_pred)

            df.to_csv(out_dir / f'{stacking_split_name}.csv')


if __name__ == '__main__':
    main()

