# UTMOS Strong Learner

Training and inference scripts for the UTMOS strong learner.

## Prerequesities

* poetry
* [WavAugment](https://github.com/facebookresearch/WavAugment)

## Setup
1. Download SSL model checkpoints from [fairseq repo](https://github.com/pytorch/fairseq).
1. Run the following commands.
```shell
cd path/to/this/repository
git submodule update --init
poetry install
ln -s path/to/dataset/ data/
```

## Preprocessing
When using phoneme encoding, you need to transcribe speech for preprocessing with the following command.
```shell
python transcribe_speech.py
```

## Training

To train the strong learner, run the following commands for each of the tracks.

Main track
```shell
python train.py dataset.data_dir=data/phase1-main/DATA
```
OOD track
```shell
python train.py dataset.data_dir=data/phase1-ood/DATA
```

## Prediction 
To predict scores with the trained strong learner, run the following command.
```shell
python predict.py +ckpt_path=outputs/${date}/${time}/train_outputs/hoge.ckpt
```