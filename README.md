<div align="center">

# Aligning Forest and Trees in Images & Long Captions<br>for Visually Grounded Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2602.02977-b31b1b.svg)](https://arxiv.org/abs/2602.02977)
[![PDF](https://img.shields.io/badge/PDF-Download-FF6F00.svg)](https://arxiv.org/pdf/2602.02977)
[![Project](https://img.shields.io/badge/Project-Page-4285F4.svg)](https://byeongju.me/CAFT/)
[![Models](https://img.shields.io/badge/Models-Drive-34A853.svg)](https://drive.google.com/drive/folders/1APFc_JniODAlWW_u2pmTknKfd40YgwH4?usp=drive_link)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Byeongju Woo](https://byeongju.me/), [Zilin Wang](https://wayne2wang.github.io/), [Byeonghyun Pak](https://byeonghyunpak.github.io/), [Sangwoo Mo](https://sites.google.com/view/sangwoomo), [Stella X. Yu](https://web.eecs.umich.edu/~stellayu/)

</div>

CAFT (**C**ross-domain **A**lignment of **F**orests and **T**rees) jointly learns local text-region alignment at intermediate representations and global image-text alignment at the final representation, discovering localized part semantics in long captions before composing them into a global image-text representation. Trained on 30M image-text pairs, CAFT achieves state-of-the-art performance on six long-text retrieval benchmarks.

---

## Environments

We provide a Docker image ([`junwha/cast:v2`](https://hub.docker.com/r/junwha/cast)) built on Ubuntu 22.04.

```bash
# pull the image (docker run below will also auto-pull it if you skip this)
docker pull junwha/cast:v2

# start a container
docker run --gpus all -it junwha/cast:v2
```

To build your own environment instead, make sure it satisfies all of the following:
- [FLAIR](https://github.com/ExplainableML/flair) (code base)
- [CAST](https://github.com/twke18/CAST) (vision backbone, adapted)
- [SFCN](https://github.com/fuy34/superpixel_fcn) (superpixel algorithm)

## Datasets

Expected data folder structure is as follows:

```
/path/to/data/
├── dreamlip3m/    # CC3M-DreamLIP webdataset shards
├── dreamlip12m/   # CC12M-DreamLIP webdataset shards
├── dreamlip15m/   # YFCC15M-DreamLIP webdataset shards
├── checkpoints/   # model checkpoints (*.pt)
└── eval/          # retrieval eval sets: dci/, docci/, share4v/, Urban1k/, imageinwords/, mscoco/, ...
```

- [Training data setup](datasets/TRAIN_DATASETS.md)
- [Evaluation data setup](datasets/EVAL_DATASETS.md)

## Checkpoints

Download checkpoints on [Google Drive](https://drive.google.com/drive/folders/1APFc_JniODAlWW_u2pmTknKfd40YgwH4?usp=drive_link). Default is **CAFT-30M**, while subsets (3M, 12M, 15M) are also released.

## Quick Demos

Coming soon (in [notebook files](src/notebooks)):
- Measure image-text similarity
- Attention map visualization
- Test-time adaptation

## Inference

Run any script under `src/` (e.g. `bash inference.sh`) after preparing the eval datasets above.

| Script | Note |
|---|---|
| `inference.sh` | Default setting |
| `inference_wholeonly.sh` | Whole score only setting |
| `inference_topk.sh` | Top-K reranking setting |

Default blends the whole score and part score with `alpha=0.3` as a weighted sum, while Top-K reranking only applies the weighted sum over the top-K whole-score candidates.

## Train

Run any script under `src/` (e.g. `bash CAFT_30M.sh`) after preparing the training data above. All models are trained with 8 A100s (80GB each). 

| Script | Data | GPU Hours |
|---|---|---|
| `CAFT_3M.sh` | DreamLIP-3M (CC3M images) | 98.6 |
| `CAFT_12M.sh` | DreamLIP-12M (CC12M images) | 340.0 |
| `CAFT_15M.sh` | DreamLIP-15M (YFCC15M images) | 476.0 |
| **`CAFT_30M.sh`** | **DreamLIP-30M (DreamLIP 3M+12M+15M)** | **915.2** |

Sample training curves: [CAFT-3M.log](src/checkpoints/CAFT-3M.log).

## Acknowledgements

This codebase builds on [FLAIR](https://github.com/ExplainableML/flair) for the training/inference code and long-text retrieval evaluation protocol, and on [CAST](https://github.com/twke18/CAST) for the vision backbone. We also use [SFCN](https://github.com/fuy34/superpixel_fcn) for superpixel segmentation, and thank [DreamLIP](https://github.com/zyf0619sjtu/DreamLIP) for the long-caption datasets used to train CAFT.

## Citation

```bibtex
@article{woo2026caft,
  title={Aligning Forest and Trees in Images \& Long Captions for Visually Grounded Understanding},
  author={Woo, Byeongju and Wang, Zilin and Pak, Byeonghyun and Mo, Sangwoo and Yu, Stella X.},
  journal={arXiv preprint arXiv:2602.02977},
  year={2026}
}
```
