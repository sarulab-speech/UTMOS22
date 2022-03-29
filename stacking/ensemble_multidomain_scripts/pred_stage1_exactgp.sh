
set -eu

for train_datatrack in phase1-ood students-wo_test  phase1-main; do
for pred_datatrack in phase1-ood phase1-main  ; do
for ssl_type in w2v_small w2v_large w2v_large2 w2v_xlsr wavlm_base wavlm_large hubert_base hubert_large; do
for method in exactgp; do
for i_cv in 0 1 2 3 4; do
    echo "${method}, ${train_datatrack}, ${ssl_type}, ${i_cv}, ${pred_datatrack}"
    poetry run python -u pred_stage1.py ${method} ${train_datatrack} ${ssl_type} ${i_cv} ${pred_datatrack}
done
done
done
done
done

echo "done"
