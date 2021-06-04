# README

R2Gen is the implementation of [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf) at EMNLP-2020. \
KG and pretrained GCN comes from RGMG and VSEGCN


## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

### RGMG's KG

link = '1-q0e7oDDIn419KlMTmGTOZWoMqJTUbpV' \
downloaded = drive.CreateFile({'id':link}) \
downloaded.GetContentFile('gcnclassifier_v2_ones3_t401v2t3_lr1e-6_e80.pth') \
**OR**

link = '10J5VwEmyOM9-I_YHyzpJaALRN36o1No4' \
downloaded = drive.CreateFile({'id':link})  \
downloaded.GetContentFile('gcnclassifier_v2_ones3_t0v1t2_lr1e-6_e80.pth') \


### VSEGCN's KG

changes to pretrained gcn download:  \
link = '1Cd_J2-tFVvRE6dMBfyJsYKW_1HPWtlHx' \
downloaded = drive.CreateFile({'id':link})  \
downloaded.GetContentFile('iuxray_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_23050521_e180.pth') \

changes to 'run_iu_xray.sh': \
--pretrained models/iuxray_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_23050521_e180.pth \
--kg_option 'vsegcn' \

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

### Two Images MIMIC-CXR
If 2 images of MIMIC-CXR is inputted, change in run_mimic_cxr.sh: \
--d_vf 2048 \
--dataset_name 'mimic_cxr_2images' 

### KG(VSEGCN)
OLD(not used anymore): \
link = '1-b6zxemYj6yoTG6rxjMW11lyZiuE0kTV' \
downloaded = drive.CreateFile({'id':link}) \
downloaded.GetContentFile('mimic_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_e10.pth') \

NEW(pls use this one) \
link = '1_5DhLPDq7bSOgLWLPO7BM-gUySqpiVCK' \
downloaded = drive.CreateFile({'id':link}) \
downloaded.GetContentFile('mimic_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_24052021_e10.pth') \



## Pretrained Language Models
### None
default, just nn.Embedding

### Glove-MIMIC
--pretrained_LM 'glove-mimic'


### BioBert
If use BioBert as pretrained Language Models:
pip install pytorch-pretrained-bert

need to change in run_iu_xray.sh:
--d_model 768 \
--rm_d_model 768 \
--pretrained_LM 'biobert'

### BioAlbert
If use BioBert as pretrained Language Models: \
pip install pytorch-pretrained-bert \
pip install transformers \
pip install sentencepiece \

need to change in run_iu_xray.sh: \
--d_model 128 \
--rm_d_model 128 \
--pretrained_LM 'bioalbert'



## Citations

If you use or extend our work, please cite our paper at EMNLP-2020.
```
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and
      Song, Yan  and
      Chang, Tsung-Hui and
      Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

## Requirements

- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`


## Download R2Gen
You can download the models we trained for each dataset from [here](https://github.com/cuhksz-nlp/R2Gen/blob/main/data/r2gen.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.


