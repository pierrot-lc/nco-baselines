CKPT_PATH=ckpts/tsp500.ckpt
python -u -m decoding.tsp \
    --data /home/pierrot-lc/GitHub/insertsp/data/tsp-500.npz \
    --ckpt $CKPT_PATH \
    --batch_size 1 \
    --sampling_steps 1 \
    --two_opt_steps 5 \
    --cycles 40 --runs 8 \
    --keep_rate 0.2 \
    --threads_over_batches 1 \
    --heatmap_dtype uint8 --topk 5000 \
    --augment_level 1 \
