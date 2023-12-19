# UTMOS: UTokyo-SaruLab MOS Prediction System

Official implementation of ["UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022"](https://arxiv.org/abs/2204.02152) accepted by <i>INTERSPEECH 2022</i>.

>**Abstract:**<br>
We present the UTokyo-SaruLab mean opinion score (MOS) prediction system submitted to VoiceMOS Challenge 2022. The challenge is to predict the MOS values of speech samples collected from previous Blizzard Challenges and Voice Conversion Challenges for two tracks: a main track for in-domain prediction and an out-of-domain (OOD) track for which there is less labeled data from different listening tests. Our system is based on ensemble learning of strong and weak learners. Strong learners incorporate several improvements to the previous fine-tuning models of self-supervised learning (SSL) models, while weak learners use basic machine-learning methods to predict scores from SSL features.
In the Challenge, our system had the highest score on several metrics for both the main and OOD tracks. In addition, we conducted ablation studies to investigate the effectiveness of our proposed methods.

🏆 Our system achieved the 1st places in 10/16 metrics at [the VoiceMOS Challenge 2022](https://voicemos-challenge-2022.github.io/)!

Demo for UTMOS is available: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sarulab-speech/UTMOS-demo)

## Quick Prediction
You can simply use a pretrained UTMOS strong leaner trained on the VoiceMOS Challenge 2022 Main Track Dataset. We support both single and batch processings in a [NISQA](https://github.com/gabrielmittag/NISQA)-like interface.

Git clone the Hugging Face repo:
```
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
cd UTMOS-demo
pip install -r requirements.txt
```

To predict the MOS of a single wav file:
```
python predict.py --mode predict_file --inp_path /path/to/wav/file.wav --out_path /path/to/csv/file.csv
```

To predict the MOS of all .wav files in a folder use:
```
python predict.py --mode predict_dir --inp_dir /path/to/wav/dir/ --bs <batchsize> --out_path /path/to/csv/file.csv
```

You can also use the [pip package](https://github.com/ttseval/utmos).

## How to use the whole functionality

### Enviornment setup

1. This repo uses poetry as the python envoirnmet manager. Install poetry following [this instruction](https://python-poetry.org/docs/#installation) first.
1. Install required python packages using `poetry install`. And enter the python enviornment with `poetry shell`. All following operations **requires** to be inside the poetry shell enviornment. 
1. Second, download necessary fairseq checkpoint using [download_strong_checkpoints.sh](fairseq_checkpoints/download_strong_checkpoints.sh) for strong and [download_stacking_checkpoints.sh](fairseq_checkpoints/download_stacking_checkpoints.sh) for stacking.
1. Next, run the following command to exclude bad wav file from main track training set.
The original data will be saved with `.bak` suffix.
```shell
python remove_silenceWav.py --path_to_dataset path-to-dataset/phase1-main/
```

## Model training
Our system predicts MOS with small errors by stacking of strong and weak learners.  
- To run training and inference with <u>a single strong learner</u>, see [strong/README.md](strong/README.md).  
- To run <u>stacking</u>, see [stacking/ensemble_multidomain_scripts/README.md](stacking/ensemble_multidomain_scripts/README.md).

If you encounter any problems regarding running the code, feel free to submit an issue. The code is not fully tested.
