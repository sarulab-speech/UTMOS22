
import os
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

SEED_CV = 0
K_CV = 5


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

    datadir = Path('../data', args.datatrack, 'DATA')
    # outdir = Path('./out/ensemble', args.datatrack, 'fold')
    outdir = Path('../out/ensemble-multidomain/fold', args.datatrack)

    os.makedirs(outdir, exist_ok=True)

    moslists = {
        "train": datadir / "sets/train_mos_list.txt",
        "val": datadir / "sets/val_mos_list.txt",
    }
    mos_lookup = {}

    wavnames = {
        'train': [],
        'val': [],
    }

    for split in ["train", "val"]:
        with open(moslists[split], "r") as fr:
            for line in fr:
                parts = line.strip().split(",")
                wavname = parts[0]
                mos = parts[1]
                mos_lookup[wavname] = mos
                wavnames[split].append(wavname)

    Kf = KFold(n_splits=K_CV, random_state=SEED_CV, shuffle=True)
    train_wavnames = np.asarray(wavnames['train'])

    for i, (cv_train_idx, cv_val_idx) in enumerate(Kf.split(train_wavnames)):

        cv_train_wavnames = train_wavnames[cv_train_idx]
        cv_val_wavnames = train_wavnames[cv_val_idx]

        write_wavnames(outdir / f'train-{i}.csv', cv_train_wavnames, mos_lookup)
        write_wavnames(outdir / f'val-{i}.csv', cv_val_wavnames, mos_lookup)
        write_wavnames(outdir / f'test-{i}.csv', wavnames['val'], mos_lookup)

    write_wavnames(outdir / f'train-all.csv', train_wavnames, mos_lookup)


if __name__ == '__main__':
    main()