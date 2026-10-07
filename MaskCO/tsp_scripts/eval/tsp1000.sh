CKPT_PATH=ckpts/tsp1000.ckpt
python -u -m decoding.tsp \
    --data tsp-1000.npz \
    --ckpt $CKPT_PATH \
    --batch_size 1 \
    --sampling_steps 2 \
    --two_opt_steps 10 \
    --cycles 20 --runs 8 \
    --keep_rate 0.1 \
    --threads_over_batches 1 \
    --heatmap_dtype uint8 \
    --topk 20000 \
    --augment_level 1 \
