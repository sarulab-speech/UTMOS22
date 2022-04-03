# Stacking of strong and weak learners

## Data split
Firstly link `data` to `../data`.
Then run the following commands.
```shell
python make_ensemble_dataset.py --datatrack phase1-main
python make_ensemble_dataset.py --datatrack phase1-ood
python make_ensemble_dataset_wotest.py --datatrack external
python make_ensemble_testphase.py --datatrack testphase-main
python make_ensemble_testphase.py --datatrack testphase-odd
```

## Feature extraction with SSL model
Place the ckpt file of the pretrained model to `../pretrained_model`.  
Then run the following command.
```shell
python extract_ssl_feature.py
```

## Converting results of strong learners for stacking
Place the respective result files to `../strong_learner_result/main1` and `../strong_learner_result/ood1`.
Then run the following commands.
```shell
python convert_strong_learner_result.py phase1-main main1
python convert_strong_learner_result.py phase1-ood ood1
python convert_strong_learner_testphase_result.py testphase-main main1
python convert_strong_learner_testphase_result.py testphase-ood ood1
```

## Stage1
For both main and OOD tracks, run the following command to perform stage1.
```shell
./run_stage1.sh
```

## Stage2 and 3 for Main track
Run the following commands.
```shell
./run_stage2-3_main.sh # Run stage 2 and 3
./pred_testphase_stage1_main.sh # Predict stage 1
./pred_testphase_stage2-3_main.sh # Predict stage 2 and 3
```

## Stage 2 and 3 for OOD track
Run the following commands.
```shell
./pred_stage1_ood.sh # Predict by cross-domain
./run_stage2-3_ood.sh # Run stage 2 and 3
./pred_testphase_stage1_ood.sh # Predict stage 1
./pred_testphase_stage2-3_ood.sh # Predict stage 2 and 3
```
