# Shape Transformers: Topology-Independent 3D Shape Models Using Transformers
This is unofficial implementation of [Shape Transformers: Topology-Independent 3D Shape Models Using Transformers](https://diglib.eg.org/handle/10.1111/cgf14468)

## Requirements
```
pytorch >= 1.10.2
openmesh >= 1.1.6
numpy >= 1.22.3
tensorboard
```
and may be more. sorry! ;)
## Dataset
The authors conducted experiments on many datasets. but this repository used only the CoMA dataset.

You can download the CoMA Dataset [here](https://coma.is.tue.mpg.de/).

After downloading, unzip the dataset like below
```
- project dir(this repo)
|-- coma_dataset/
  |-- FaceTalk_170725_00137_TA/
  |-- FaceTalk_170725_00137_TA/
  '-- ...
|-- net/
|-- networks/
|- train.py
|- pred.py
'- readme.md
```

## Train
```
python train.py
```

## Test
```
python test.py
```
Work on progress. currently, it's just testing network using untrained facial shapes.


May be you can try sparse input by sampling down vertices like described in the paper.
## Misc.
My implementation may differ from described in the paper. so any comments are welcome.

This repository contains codes from [XCiT](https://github.com/facebookresearch/xcit), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models).

Codes from the other repositories are subject to different licenses applied by each author.




##  BibTex

```
@article {10.1111:cgf.14468,
journal = {Computer Graphics Forum},
title = {{Shape Transformers: Topology-Independent 3D Shape Models Using Transformers}},
author = {Chandran, Prashanth and Zoss, Gaspard and Gross, Markus and Gotardo, Paulo and Bradley, Derek},
year = {2022},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14468}
}
```