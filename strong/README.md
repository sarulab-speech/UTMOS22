# prerequesities

* poetry
* [WavAugment](https://github.com/facebookresearch/WavAugment)

# setup
1. download ssl model file from fairseq webpage
1. run the commands below
```
cd path/to/this/repository
git submodule update --init
poetry install
ln -s path/to/dataset/ data/
```

# transcribing speech
```
python transcribe_speech.py
```

# training

main_track
```
python train.py dataset.data_dir=data/phase1-main/DATA
```
ood_track
```
python train.py dataset.data_dir=data/phase1-ood/DATA
```

# finetuning from ckpt
```
python train.py dataset.data_dir=same_as_above train.finetuned_checkpoint=path_to_pt_file
```
