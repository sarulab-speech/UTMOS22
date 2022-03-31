
set -eu

# ssl_types="w2v_small w2v_large w2v_large2 w2v_xlsr wavlm_base wavlm_large hubert_base hubert_large"
ssl_types="w2v_small w2v_large wavlm_base wavlm_large hubert_base hubert_large"

for datatrack in phase1-ood  phase1-main external-wo_test; do
for method in ridge linear_svr kernel_svr rf lightgbm exactgp; do
for ssl_type in ${ssl_types}; do
for i_cv in 0 1 2 3 4; do
    echo "${datatrack}, ${method}, ${ssl_type}, ${i_cv}"
    python -u run_stage1.py ${method} ${datatrack} ${ssl_type} ${i_cv}
done
done
done
done

echo "done"
