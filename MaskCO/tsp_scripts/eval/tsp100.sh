CKPT_PATH=ckpts/tsp100.ckpt
python -u -m decoding.tsp \
    --data /home/pierrot-lc/GitHub/insertsp/data/tsp-100.npz \
    --ckpt $CKPT_PATH \
    --batch_size 1 \
    --sampling_steps 1 \
    --two_opt_steps 1 \
    --cycles 40 --runs 8 \
    --keep_rate 0.2 \
    --threads_over_batches 1 \
    --augment_level 1 \
