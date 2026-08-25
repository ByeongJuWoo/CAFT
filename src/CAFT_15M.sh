torchrun --nnode 1 --nproc_per_node 8 -m main \
    --model CAFT-B \
    --train-data '/path/to/data/dreamlip15m/yfcc15m-{000000..001804}.tar' \
    --inference-mode caft \
    --dci-retrieval-dir /path/to/data/eval/dci \
    --retrieval-dci \
    --val-frequency 2 \
    --train-num-samples  14066050 \
    --train-dataset-type webdataset  \
    --lr 1.67e-4 \
    --wd 0.5 \
    --warmup 2000 \
    --epochs 32  \
    --num-sampled-captions 8 \
    --log-every-n-steps 200 \
    --caption-sampling-mode caft_sampling \
    --batch-size 256 \
    --eval-img-batch-size 128 \
    --precision amp \
    --workers 12 \
    --save-frequency 4 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.5 \
    --eps 1e-6 \
    --grad-clip-norm 1.0 


    



