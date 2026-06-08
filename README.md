# Geometry-Aware Dataset Condensation for Diffusion Model Training

## Preparation
### Environment
For convenience, you can create the Conda virtual environment through:

```
conda env create -f environment.yaml
conda activate GADC
```

### Downloading Dataset
Download ImageNet-1K from ```https://www.image-net.org/challenges/LSVRC/index.php``` and locate it at:

```
../imagenet/
```

### Downloading Reference Batches
Download reference batches for evaluation from ```https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz``` and ```https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz```, then locate them at:

```
evaluation/reference/
```

## Data Selection
Run the following command:
```
cd selection
sh run.sh
```

## Diffusion Model Training
Run the following command:
```
cd training
sh run.sh
```

## Image Generation and Evaluation
Run the following command:
```
cd evaluation
sh run.sh
sh test.sh
```

Then you can get the FID, Inception Score, Precision and Recall.

## BibTeX
```
@article{cui2026geometry,
  title={Geometry-Aware Dataset Condensation for Diffusion Model Training},
  author={Cui, Xiao and Qin, Yulei and Zhu, Mo and Zhou, Wengang and Li, Hongsheng and Li, Houqiang},
  journal={arXiv preprint arXiv:2606.05883},
  year={2026}
}
```

## Acknowledgement

Thanks to Rui Huang et al. for their nice work, **Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective**！
