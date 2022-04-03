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

def main():

    args = get_arg()

    phase = 'main'
    in_dir = Path('../strong_learner_result') / args.learner
    out_base_dir = Path(f'../out/ensemble-multidomain/stage1/') \
                    / args.datatrack / args.learner

    k_cv = 3 if args.datatrack == 'testphase-ood' else 5
    answer_file_tag = 'answer-ood' if args.datatrack == 'testphase-ood' else 'answer-main'

    for i_cv in range(k_cv):
        out_dir = out_base_dir / str(i_cv) / f'pred-{args.datatrack}'
        os.makedirs(out_dir, exist_ok=True)
        print(out_dir)

        in_path = in_dir / f'{answer_file_tag}.csvtest_{i_cv}'

        df_pred = pd.read_csv(in_path, header=None,
                    names=['wavbase', 'pred_mos'])
        df_pred["wavname"] = df_pred["wavbase"] + ".wav"
        df_pred['true_mos'] = -99.0
        df = df_pred.set_index('wavname')[['pred_mos', 'true_mos']]

        df.to_csv(out_dir / f'test.csv')


if __name__ == '__main__':
    main()

