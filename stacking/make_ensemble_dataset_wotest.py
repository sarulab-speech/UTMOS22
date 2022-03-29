
import os
import argparse
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import KFold

SEED_CV = 0
K_CV = 5

NUM_VAL_FILES = 100

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datatrack", type=str, required=True, help="phase1-main or phase1-ood",
        default='phase1-main',)
    return parser.parse_args()

def write_wavnames(outpath, wavnames, mos_lookup):

    with open(outpath, 'w') as f:
        for wavname in sorted(wavnames):
            if wavname == 'sys4bafa-uttc2e86f6.wav':
                print('Skip sys4bafa-uttc2e86f6')
                continue
            print('{},{}'.format(wavname, mos_lookup[wavname]), file=f)


def main():
    args = get_arg()

    datadir = Path('./data', args.datatrack, 'DATA')
    # outdir = Path('./out/ensemble', args.datatrack + '-wo_test', 'fold')
    outdir = Path('./out/ensemble-multidomain/fold', args.datatrack + '-wo_test')

    os.makedirs(outdir, exist_ok=True)

    if args.datatrack == 'lancers':
        moslist_files = ['./lancers_mos_list.txt']
    elif args.datatrack == 'students':
        moslist_files = ['./data/additional/student_mos_list.txt']
    elif args.datatrack == 'students-ex':
        moslist_files = ['./data/additional/student-ex_mos_list.txt']
    else:
        moslist_files = [
            datadir / "sets/train_mos_list.txt",
            datadir / "sets/val_mos_list.txt",
        ]
    mos_lookup = {}

    # wavnames = {
    #     'train': [],
    #     'val': [],
    # }
    wavnames = []

    for mos_file in moslist_files:
        with open(mos_file, "r") as fr:
            for line in fr:
                parts = line.strip().split(",")
                wavname = parts[0]
                mos = parts[1]
                mos_lookup[wavname] = mos
                wavnames.append(wavname)

    train_wavnames = np.asarray(wavnames)

    rng = np.random.default_rng(SEED_CV)

    test_wavnames = list(sorted(rng.permutation(train_wavnames)[:NUM_VAL_FILES]))

    Kf = KFold(n_splits=K_CV, random_state=SEED_CV, shuffle=True)


    for i, (cv_train_idx, cv_val_idx) in enumerate(Kf.split(train_wavnames)):

        cv_train_wavnames = train_wavnames[cv_train_idx]
        cv_val_wavnames = train_wavnames[cv_val_idx]

        write_wavnames(outdir / f'train-{i}.csv', cv_train_wavnames, mos_lookup)
        write_wavnames(outdir / f'val-{i}.csv', cv_val_wavnames, mos_lookup)
        write_wavnames(outdir / f'test-{i}.csv', test_wavnames, mos_lookup)

    write_wavnames(outdir / f'train-all.csv', train_wavnames, mos_lookup)


if __name__ == '__main__':
    main()