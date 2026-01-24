cd text
python text_embedding.py
cd ..

cd visual
python extract_visual_info.py \
    --max-train-steps=400000     --report-to="wandb" \
    --allow-tf32     --mixed-precision="fp16"     --seed=0   \
    --data-dir="../imagenet/train/"   \
    --output-dir="exps"     --batch-size=32   \
    --path-type="linear"     --prediction="v"  \
    --weighting="uniform"     --model="SiT-XL/2"  \
    --checkpointing-steps=50000     --loss-cfg-path="configs/l1_lpips_kl_gan.yaml"  \
    --vae="f8d4"     --vae-ckpt="sdvae-f8d4.pt"  \
    --enc-type="dinov2-vit-b"     --proj-coeff=0.5     --encoder-depth=8   \
    --vae-align-proj-coeff=1.5     --bn-momentum=0.1   \
    --exp-name="sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
cd ..

python -m torch.distributed.run --nproc-per-node=8 train_dit.py --select_num 10000 --interval 20 \
    --embedding-root visual/test/  \
    --embedding-json visual/test/total_json.json \
    --target-mean 0.0 --image-size 256 --epochs 1300
