# Usage

## Training
1. ./opt_stage1_all.sh  # Optimize hyper parameters for stage 1
1. ./run_stage1_exactgp.sh  # Run GPR training using GPU
1. ./run_stage1_other.sh  # Run training of weak learners except GPR
1. ./run_stage2-end_opt.sh  # Run stage 2 & 3 training with hyperparameter optimization and calc training results

## Prediction
1. ./pred_testphase_stage1_all.sh # Predict stage 1 outputs
1. ./pred_testphase_stage2-end.sh # Predict stage 2 & 3 outputs

