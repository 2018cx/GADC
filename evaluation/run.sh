python -m torch.distributed.run --nproc-per-node=8 --master-port=29701 sample_test.py --sample-dir "test"\
    --num-fid-samples 50000 --ckpt ""

