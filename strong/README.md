# UTMOS Strong Learner

Training and inference scripts for the UTMOS strong learner.

## Prerequesities

* poetry
* [WavAugment](https://github.com/facebookresearch/WavAugment)

## Pretrained model
Pretrained **UTMOS strong** models for the main and OOD tracks are available.  
For the model details, refer to the [paper](https://arxiv.org/abs/2204.02152).

- [Main](https://drive.google.com/drive/folders/1U4XQze8mJqV4TRMwTcY6T247RpmU5hRg?usp=sharing)
- [OOD](https://drive.google.com/drive/folders/1dPlV92fyKY1arei7TcU2ZFB-wZkYhIqK?usp=sharing)

Note that each of the above directory contains pretrained models obtained with five different random seeds.

## Setup
1. Download SSL model checkpoints from [fairseq repo](https://github.com/pytorch/fairseq).
1. Run the following commands.
```shell
cd path/to/this/repository
poetry install
cd strong/
ln -s path/to/dataset/ data/
poetry shell
```

## Preprocessing
The phoeneme transcription is alreday present in this repository.
You can also perform the transcribe by your own!
```shell
cd strong/
python transcribe_speech.py
```

## Training

To train the strong learner, run the following commands for each of the tracks.

Main track
```shell
cd strong/
python train.py dataset.use_data.ood=False dataset.use_data.external=False
```
OOD track
```shell
cd strong/
python train.py train.model_selection_metric=val_SRCC_system_ood
```

## Prediction 
To predict scores with the trained strong learner, run the following command.
```shell
python predict.py +ckpt_path=outputs/${date}/${time}/train_outputs/hoge.ckpt
```
To perform prdiction with the pretrained model, run the following command.

```shell
python predict.py +ckpt_path=outputs/${date}/${time}/train_outputs/hoge.ckpt +paper_weights=True
```
