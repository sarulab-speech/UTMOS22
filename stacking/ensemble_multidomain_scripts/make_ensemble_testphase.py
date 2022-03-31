
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
        "--datatrack", type=str, required=True, help="testphase-main or testphase-ood",
        default='testphase-main',)
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

    assert args.datatrack in ['testphase-main', 'testphase-ood']

    datadir = Path('../data', args.datatrack, 'DATA')
    outdir = Path('../out/ensemble-multidomain/fold', args.datatrack)

    os.makedirs(outdir, exist_ok=True)

    moslist_file = datadir / "sets/test.scp"
    mos_lookup = {}

    wavnames = []

    with open(moslist_file, "r") as fr:
        for line in fr:
            parts = line.strip().split(",")
            wavname = parts[0]
            mos = -99.0 # dummy
            mos_lookup[wavname] = mos
            wavnames.append(wavname)

    write_wavnames(outdir / f'test-0.csv', wavnames, mos_lookup)


if __name__ == '__main__':
    main()