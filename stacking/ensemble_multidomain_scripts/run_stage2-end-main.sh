
set -eu

datatrack=phase1-main
feat_type=main-strong1-weak48

poetry  run python -u collect_stage1_result.py ${datatrack} ${feat_type}

for method in ridge linear_svr kernel_svr rf lightgbm exactgp; do
for i_cv in 0 1 2 3 4; do
    echo "Run stage2: ${method}, ${datatrack}, ${feat_type}, ${i_cv}"
    poetry run python -u run_stage2.py ${method} ${datatrack} ${feat_type} ${i_cv}
done
done

echo "Collect stage2 data."
poetry  run python -u collect_stage2_result.py ${datatrack} ${feat_type}

for method in ridge; do
for i_cv in 0 1 2 3 4; do
    echo "Run stage3: ${method} ${datatrack}, ${feat_type}, ${i_cv}"
    poetry run python -u run_stage3.py ${method} ${datatrack} ${feat_type} ${i_cv}
done
done

echo "Calculate result."

poetry run python -u calc_result.py ${datatrack} ${feat_type}

echo "done"
