torchrun --nnode 1 --nproc_per_node 8 -m main \
    --model ViT-B-16-FLAIR \
    --train-data '/path/to/data/dreamlip3m/cc3m-train-{0000..0575}.tar' \
    --inference-mode flair \
    --val-frequency 2 \
    --train-num-samples  2823423 \
    --train-dataset-type webdataset  \
    --lr 5e-4 \
    --wd 0.5 \
    --warmup 2000 \
    --epochs 32  \
    --use-flair-loss \
    --add-mps-loss \
    --num-sampled-captions 8 \
    --log-every-n-steps 200 \
    --caption-sampling-mode diverse_sampling \
    --batch-size 256 \
    --precision amp \
    --workers 12 \
    --save-frequency 4 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.5 \
    --eps 1e-8 \
    


    



