
set -eu

ssl_types="w2v_small w2v_large w2v_large2 w2v_xlsr wavlm_base wavlm_large hubert_base hubert_large"

for method in ridge lightgbm linear_svr kernel_svr; do
for datatrack in phase1-ood phase1-main students-wo_test ; do
for ssl_type in ${ssl_types}; do
    echo "${datatrack}, ${ssl_type}, ${method}"
    poetry run python -u opt_stage1.py ${method} ${datatrack} ${ssl_type}
done
done
done

echo "done"

