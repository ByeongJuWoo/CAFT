# Data Preparation for Text-Image Retrieval

Note: Our dataset setup mostly follows FLAIR's, with a few differences noted below.

### Annotation Files
We pre-processed and unified the annotations for various datasets to be in `.json` format to standardize them. These annotation files are stored under `datasets/` directory of this repo. To use our inference code properly, you should also use the same annotation files, the detailed instructions are as follows:

### Overall structure
```
/path/to/data/eval/
├── docci/
│   ├── images/
│   └── annotations/
├── imageinwords/
│   ├── dci/
│   ├── docci/
│   ├── docci_aar/
│   ├── DOCCI_Test/
│   ├── IIW-400/
│   └── DCI_Test/
├── share4v/
│   ├── sa_000000/
│   ├── sa_000001/
│   └── ...
├── dci/
│   └── densely_captioned_images/
│       ├── annotations/
│       ├── photos/
│       └── splits.json
├── Urban1k/
│   ├── images/
│   └── caption/
├── mscoco/
│   ├── images/val2017/
│   └── annotations/
└── flickr30k-images/
```

### Datasets list:
- [DOCCI](#docci)
- [IIW](#IIW)
- [ShareGPT4v](#share4v)
- [DCI](#DCI)
- [Urban1k](#urban1k)
- [MSCOCO](#coco)
- [FLICKR30K](#flickr)


### <span id ='docci'> DOCCI dataset
```
$docci/
|––  images/
|––––  test_01427.jpg
|––––  test_01428.jpg
|––––  ...
|––  annotations/
|–––– docci_descriptions.jsonline
```
Step 1. Download [DOCCI Images](https://storage.googleapis.com/docci/data/docci_images.tar.gz), unzip them under `docci/images/`, note that we only need the 5K test images here

Step 2. Download [docci_descriptions.jsonlines](https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines) from the [official DOCCI downloads](https://google.github.io/docci/#downloads) and place it at `docci/annotations/docci_descriptions.jsonline` (note the singular `.jsonline` — rename the file when you place it, that's the exact name our code expects)

### <span id ='iiw'> IIW dataset

```
$imageinwords/
|–– dci/
|–– docci/
|–– docci_aar/
|–– DOCCI_Test/
|–––– data.jsonl
|–– IIW-400/
|–––– data.jsonl
|–– DCI_Test/
|–––– data.jsonl
```

**Download human annotated data following [IIW](https://github.com/google/imageinwords/tree/main/datasets)**:

Step 1: Download [DCI](https://github.com/facebookresearch/DCI) images to `imageinwords/dci/`

Step 2: Download DOCCI images and AAR images from [DOCCI](https://google.github.io/docci/#downloads), unzip to `imageinwords/docci/` and `imageinwords/docci_aar/` respectively

Step 3: Download the `DOCCI_Test/`, `IIW-400/`, `DCI_Test/` folders as-is from the [IIW repo's `datasets/`](https://github.com/google/imageinwords/tree/main/datasets) directory and place them under `imageinwords/` — each keeps its own `data.jsonl`, no repackaging needed


### <span id ='share4v'> ShareGPT4v dataset

```
$share4v/
|–– sa_000000/
|–––– images/
|–––––– sa_1.jpg
|–––––– sa_2.jpg
|–––––– ...
|–– sa_000001/
|–– ...
|–– share4v_sam_10k.json
```

Step 1. Download tar files from [SA-1B](https://huggingface.co/datasets/sailvideo/SA-1B) to `share4v/`

Step 2. Unzip all tar files

Step 3. The annotation file (the top 10k samples resaved from [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main), same file FLAIR uses) is bundled in this repo at [`datasets/share4v/share4v_sam_10k.json`](share4v/share4v_sam_10k.json) — move it to `share4v/share4v_sam_10k.json` under your data root


### <span id ='dci'> DCI dataset

```
$dci/
|–– densely_captioned_images/
|–––– annotations/
|–––– photos/
|–––– splits.json

```

**Download data following [DCI](https://github.com/facebookresearch/DCI)**:

Step 1. Download [dci.tar.gz](https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz) and unzip the file in `dci/densely_captioned_images` 

Step 2. Download the archive sa_000138.tar and extract the images to the `dci/densely_captioned_images/photos folder`.


### <span id ='urban1k'> Urban1k dataset
```
$Urban1k/
|––  images/
|––––  221.jpg
|––––  222.jpg
|––––  ...
|––  caption/
|––––  221.txt
|––––  222.txt
|––––  ...
```
Step 1. Download [Urban1K](https://huggingface.co/datasets/BeichenZhang/Urban1k) and unzip it

Step 2. Place the image folder as `Urban1k/images/` and the caption folder (one `.txt` per image, same basename) as `Urban1k/caption/` — this is the dataset's native format, no extra processing needed

### <span id ='coco'> MSCOCO dataset
```
$coco/
|–– images/
|–––– val2017/
|–––––– 000000134722.jpg
|–––––– 000000177015.jpg
|–––––– ...
|–– annotations/
|–––– captions_val2017.json
```
Step 1. Download validation images from [COCO 2017 Val Images](https://cocodataset.org/#download), unzip them to `coco/images/val2017`

Step 2. Download the 2017 Val annotations, place it under `coco/annotations/captions_val2017.json`

### <span id ='flickr'> FLCIKR30K dataset
```
$flickr30k-images/
|––  2217728745.jpg 
|––  2217728745.jpg
|––  ...
|––  flickr30k_val.json
|––  flickr30k_test.json
```
Step 1. Download  [flickr30k dataset](https://huggingface.co/datasets/nlphuji/flickr30k), unzip them under `flickr30k-images/`, all the images and annotations files will be structured as above
