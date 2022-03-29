# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

## Get a pretrained model and run inference; generate an answers.txt file that
## can be submitted to the challenge server.

import os
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path of your DATA/ directory"
    )
    args = parser.parse_args()
    DATADIR = args.datadir

    ## 1. download the base model from fairseq
    if not os.path.exists("fairseq/wav2vec_small.pt"):
        os.system("mkdir -p fairseq")
        os.system(
            "wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P fairseq"
        )
        os.system(
            "wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P fairseq/"
        )

    ## 2. download the finetuned checkpoint
    if not os.path.exists("pretrained/ckpt_w2vsmall"):
        os.system("mkdir -p pretrained")
        os.system("wget https://www.dropbox.com/s/brjocxwaiemayga/ckpt_w2vsmall.tar.gz")
        os.system("tar -zxvf ckpt_w2vsmall.tar.gz")
        os.system("mv ckpt_w2vsmall pretrained/")
        os.system("rm ckpt_w2vsmall.tar.gz")
        os.system("cp fairseq/LICENSE pretrained/")

    ## 2. run inference
    os.system(
        "python predict.py --fairseq_base_model fairseq/wav2vec_small.pt --outfile answer_main.txt --finetuned_checkpoint pretrained/ckpt_w2vsmall --datadir "
        + DATADIR
    )


main()
