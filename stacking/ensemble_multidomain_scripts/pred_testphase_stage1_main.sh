
set -eu

# ssl_types="w2v_small w2v_large w2v_large2 w2v_xlsr wavlm_base wavlm_large hubert_base hubert_large"
ssl_types="w2v_small w2v_large wavlm_base wavlm_large hubert_base hubert_large"

for train_datatrack in phase1-main; do
for pred_datatrack in testphase-main; do
for ssl_type in ${ssl_types}; do
for method in ridge linear_svr kernel_svr rf lightgbm exactgp ; do
for i_cv in 0 1 2 3 4; do
    echo "${method}, ${train_datatrack}, ${ssl_type}, ${i_cv}, ${pred_datatrack}"
    python -u pred_testphase_stage1.py ${method} ${train_datatrack} ${ssl_type} ${i_cv} ${pred_datatrack}
done
done
done
done
done

echo "done"
