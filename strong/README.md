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
