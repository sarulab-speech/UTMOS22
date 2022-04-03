
set -eu

datatrack=phase1-main
feat_type=main-strong1-weak48-opt

poetry  run python -u collect_stage1_result.py ${datatrack} ${feat_type}

for method in ridge linear_svr kernel_svr lightgbm; do
    echo "Optimize hyperparameter for stage2: ${datatrack}, ${feat_type}, ${method}"
    poetry run python -u opt_stage2.py ${method} ${datatrack} ${feat_type}
done


for method in exactgp ridge linear_svr kernel_svr rf lightgbm; do
for i_cv in 0 1 2 3 4; do
    echo "Run stage2: ${method}, ${datatrack}, ${feat_type}, ${i_cv}"
    poetry run python -u run_stage2.py --use_opt ${method} ${datatrack} ${feat_type} ${i_cv}
done
done

echo "Collect stage2 data."
poetry  run python -u collect_stage2_result.py ${datatrack} ${feat_type}


for method in ridge; do
    echo "Optimize hyperparameter for stage3: ${datatrack}, ${feat_type}, ${method}"
    poetry run python -u opt_stage3.py ${method} ${datatrack} ${feat_type}
done

for method in ridge; do
for i_cv in 0 1 2 3 4; do
    echo "Run stage3: ${method} ${datatrack}, ${feat_type}, ${i_cv}"
    poetry run python -u run_stage3.py --use_opt ${method} ${datatrack} ${feat_type} ${i_cv}
done
done


echo "Calculate result."

poetry run python -u calc_result.py ${datatrack} ${feat_type}

echo "done"
