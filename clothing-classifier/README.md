
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/SamirGouda/YOLOX-Clothing_Classifier/tree/main/clothing-classifier">
    <img src="images/pl.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Clothing Classifier</h3>

  <p align="center">
    classification of clothing images of 10 categories
    
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#approach">Approach</a></li>
    <li><a href="#data-handling">Data Handling</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#samples">SAMPLES</a></li>
    <li><a href="#receptive-field">Receptive Field</a></li>
    <li><a href="#gflops">GFlops</a></li>
    <li><a href="#veridict">Veridict</a></li>
    <li><a href="#whats-next">What's Next</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Introduction

<!-- [![Product Name ScreenShot][product-screenshot]](https://example.com) -->

This repo trains a dnn model to classify images of clothing, like dress, hat, shirt and pants.


## Prerequisites

1. [pytorch](https://pytorch.org) install with anaconda "handles cuda and other conflicts"

```sh
# create virtual environment (pytorch has many conflicts, it's better to install it in virtual env)
conda create --name myenv python=3.9
# install pytorch-cuda
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch=1.12.1 -c conda-forge -y
```

2. [pytorch-lightning](https://www.pytorchlightning.ai)
```
conda install -c conda-forge pytorch-lightning
```

3. other requirements
```
pip install -r requirements.txt
```

## Approach

1. at first, i read [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf) paper to refresh my memory about image classification, then i looked for pretrained model used for clothing classification, but mostly i found models about clothing segmentation and posing detection, so i decided to train a model with availabe dataset, so i looked for clothing dataset and i found [clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small)

2. i performed data analysis on the dataset (images distribution and class counts) to study imbalancing, and better help me build the dataset and the dataloader for ex. `WeightedRandomSampler` were used to mitigate data imbalanced classes
```
python analyze_dataset.py --input-dir clothing-dataset-small
```

3. i built from scratch a pytorch training scripts with [pytorch-lightning](https://www.pytorchlightning.ai) wrapper and these scripts can be used to train or finetune any pytorch model with ease, the code is minimal and easily read and well documented. the script supports some SOTA techniques and algorithms to improve model convergence and reduce training time for example `amp`, `swa` and `lr_tuning`, it also supports `tensorboard` or online web app `wandb`

4. i choose `mobilenet_v3_small` model from [pytorch-zoo](https://github.com/rwightman/pytorch-image-models) due to its small architecture and minimal GFLOPS (less computations) and it can be easily deployed to mobile phones cpus

5. i started tuning the model params, `lr`, `batch_size` and found best values for optimum training
```
python finetune_vision_mdl_pytorch.py --conf conf/mobilenet.yaml \
  --use-gpu --tune
```

6. model finetuned with `clothing-dataset-small` for `40` epochs with [conf](conf/mobilenet.yaml)
```
python finetune_vision_mdl_pytorch.py --conf conf/mobilenet.yaml \
  --use-gpu --exp-dir exp/pytorch/mobilenet_v3_small --run-id "v1"
```

7. i did model analysis like calc `FLOPs` and model `receptive field`
```
# find flops per layer
python calc_model_flops.py --conf conf/mobilenet.yaml --log results/mobilenet_v3_small.flops.analysis
# analyze receptive field
python analyze_model_receptive_field.py --conf conf/mobilenet.yaml --log results/mobilenet_v3_small.rf.analysis
```

<!-- Data -->
## Data Handling
[![Dataset][dataset]](https://github.com/alexeygrigorev/clothing-dataset-small)

the dataset consists of **3781** images seperated into `train`, `val` and `test` as follows

|subset|num-of-imgs|
|-|-|
|train| 3068|
|val| 341|
|test|372|


the images are taken for clothes on sheets and not worn by people. images are RGB `400x534` px
i created a dataset that augments the training images with simple `torchvision augmentations` and the test images are just normalized to `224x224`
i used `pytorch_lightning LightningDataModule` to wraps the datasets and the dataloader, also i handled class imbalancing

## Results

**note**: weights of `resnet50` is not uploaded because it's very large `300MB`

i finetuned both `mobilenet_v3_small` and `resnet50` to examine effect of model size on system performance.
i choose `confussion matrix` and `accuracy` and `f1-scores` to be the metrics for model evaluation because confussion matrix tells me how the model mis-classify class with another, and the accuracy gives rough indication of how the model performs, and f1-score `macro` gives unweighted mean for f1-score per class

- `mobilenet_v3_small`

|   Class    |     accuracy    | precision | recall |   f1  | num_samples |
|-|-|-|-|-|-|
|    test    | 85.22%[317/372] |     -     |   -    |   -   |     372     |
|  weighted  |        -        |   86.14   | 85.22  | 85.28 |      -      |
|   micro    |        -        |   85.22   | 85.22  | 85.22 |      -      |
|   macro    |        -        |   82.02   | 83.58  | 82.06 |      -      |
|   dress    |        -        |   51.85   | 93.33  | 66.67 |      15     |
|    hat     |        -        |   83.33   | 83.33  | 83.33 |      12     |
| longsleeve |        -        |   83.58   | 77.78  | 80.58 |      72     |
|  outwear   |        -        |   85.37   | 92.11  | 88.61 |      38     |
|   pants    |        -        |   91.30   | 100.00 | 95.45 |      42     |
|   shirt    |        -        |   73.91   | 65.38  | 69.39 |      26     |
|   shoes    |        -        |   98.63   | 98.63  | 98.63 |      73     |
|   shorts   |        -        |   81.48   | 73.33  | 77.19 |      30     |
|   skirt    |        -        |   81.82   | 75.00  | 78.26 |      12     |
|  t-shirt   |        -        |   88.89   | 76.92  | 82.47 |      52     |


we can see that the model suffers with `shirts` and `dress`

the confusion matrix shows the model mostly confusses `t-shirt` as `longsleeve`

| true/predicted | dress | hat | longsleeve | outwear | pants | shirt | shoes | shorts | skirt | t-shirt |
|-|-|-|-|-|-|-|-|-|-|-|
|     dress      |   14  |  0  |     0      |    0    |   0   |   0   |   0   |   0    |   0   |    1    |
|      hat       |   1   |  10 |     0      |    0    |   0   |   0   |   1   |   0    |   0   |    0    |
|   longsleeve   |   1   |  1  |     56     |    3    |   0   |   4   |   0   |   3    |   0   |    4    |
|    outwear     |   1   |  0  |     0      |    35   |   0   |   2   |   0   |   0    |   0   |    0    |
|     pants      |   0   |  0  |     0      |    0    |   42  |   0   |   0   |   0    |   0   |    0    |
|     shirt      |   5   |  0  |     2      |    2    |   0   |   17  |   0   |   0    |   0   |    0    |
|     shoes      |   0   |  0  |     0      |    0    |   0   |   0   |   72  |   1    |   0   |    0    |
|     shorts     |   0   |  0  |     2      |    0    |   4   |   0   |   0   |   22   |   2   |    0    |
|     skirt      |   1   |  1  |     0      |    0    |   0   |   0   |   0   |   1    |   9   |    0    |
|    t-shirt     |   4   |  0  |     7      |    1    |   0   |   0   |   0   |   0    |   0   |    40   |



- `resnet50`

|   Class    |     accuracy    | precision | recall |   f1  | num_samples |
|-|-|-|-|-|-|
|    test    | 88.44%[329/372] | 11.56%[43/372] |     -     |   -    |   -   |     372     |
|  weighted  |        -        |       -        |   90.19   | 88.44  | 88.76 |      -      |
|   micro    |        -        |       -        |   88.44   | 88.44  | 88.44 |      -      |
|   macro    |        -        |       -        |   86.73   | 86.61  | 85.73 |      -      |
|   dress    |        -        |       -        |   48.00   | 80.00  | 60.00 |      15     |
|    hat     |        -        |       -        |   100.00  | 83.33  | 90.91 |      12     |
| longsleeve |        -        |       -        |   81.82   | 87.50  | 84.56 |      72     |
|  outwear   |        -        |       -        |   92.11   | 92.11  | 92.11 |      38     |
|   pants    |        -        |       -        |   95.45   | 100.00 | 97.67 |      42     |
|   shirt    |        -        |       -        |   90.00   | 69.23  | 78.26 |      26     |
|   shoes    |        -        |       -        |   97.30   | 98.63  | 97.96 |      73     |
|   shorts   |        -        |       -        |   96.30   | 86.67  | 91.23 |      30     |
|   skirt    |        -        |       -        |   68.75   | 91.67  | 78.57 |      12     |
|  t-shirt   |        -        |       -        |   97.56   | 76.92  | 86.02 |      52     |

`resnet50` only outperform `mobilenet_v3_small` by `3%` 

| true/predicted | dress | hat | longsleeve | outwear | pants | shirt | shoes | shorts | skirt | t-shirt |
|-|-|-|-|-|-|-|-|-|-|-|
|     dress      |   12  |  0  |     1      |    0    |   0   |   0   |   0   |   0    |   1   |    1    |
|      hat       |   0   |  10 |     0      |    0    |   0   |   0   |   2   |   0    |   0   |    0    |
|   longsleeve   |   4   |  0  |     63     |    2    |   0   |   2   |   0   |   0    |   1   |    0    |
|    outwear     |   0   |  0  |     3      |    35   |   0   |   0   |   0   |   0    |   0   |    0    |
|     pants      |   0   |  0  |     0      |    0    |   42  |   0   |   0   |   0    |   0   |    0    |
|     shirt      |   4   |  0  |     2      |    0    |   0   |   18  |   0   |   0    |   2   |    0    |
|     shoes      |   0   |  0  |     0      |    1    |   0   |   0   |   72  |   0    |   0   |    0    |
|     shorts     |   0   |  0  |     1      |    0    |   2   |   0   |   0   |   26   |   1   |    0    |
|     skirt      |   0   |  0  |     0      |    0    |   0   |   0   |   0   |   1    |   11  |    0    |
|    t-shirt     |   5   |  0  |     7      |    0    |   0   |   0   |   0   |   0    |   0   |    40   |

## SAMPLES

- in-domain samples (from same distribution)
[![test-results][4]]()
[![test-results][5]]()
[![test-results][6]]()
[![test-results][7]]()
[![test-results][8]]()

- out-domain samples (N/A)

## Receptive Field

for single path networks the receptive field is calculated recursively as follows


[![receptive-field-equation][2]](https://distill.pub/2019/computing-receptive-fields/)



for example in next fig. 
[![example][1]](https://distill.pub/2019/computing-receptive-fields/)
```
rf = (2)1+ (1)2+ (2)2*1+ 1 = 9
```
since i used large networks, i did not implement the code that estimates the rf and used [torchscan](https://github.com/frgfm/torch-scan) to find estimates of rv

- `mobilenet_v3_small`

overall receptive field=

- `resnet50`

overall receptive field=3499



## GFLOPS

- `mobilenet_v3_small`

|    Layer     |         Type         |  MACCs   |  Params  |
|-|-|-|-|
|   features   |      Sequential      | 60.835M  | 927.008K |
|  features.0  | Conv2dNormActivation |  6.222M  | 464.000B |
|  features.1  |   InvertedResidual   |  1.706M  | 744.000B |
|  features.2  |   InvertedResidual   |  6.680M  |  3.864K  |
|  features.3  |   InvertedResidual   |  4.560M  |  5.416K  |
|  features.4  |   InvertedResidual   |  3.461M  | 13.736K  |
|  features.5  |   InvertedResidual   |  5.425M  | 57.264K  |
|  features.6  |   InvertedResidual   |  5.425M  | 57.264K  |
|  features.7  |   InvertedResidual   |  2.915M  | 21.968K  |
|  features.8  |   InvertedResidual   |  3.718M  | 29.800K  |
|  features.9  |   InvertedResidual   |  4.774M  | 91.848K  |
| features.10  |   InvertedResidual   |  6.564M  | 294.096K |
| features.11  |   InvertedResidual   |  6.564M  | 294.096K |
| features.12  | Conv2dNormActivation |  2.822M  | 56.448K  |
|   avgpool    |  AdaptiveAvgPool2d   | 28.800K  |  0.000B  |
|  classifier  |      Sequential      |  1.614M  |  1.616M  |
| classifier.0 |        Linear        | 589.824K | 590.848K |
| classifier.1 |      Hardswish       |  0.000B  |  0.000B  |
| classifier.2 |       Dropout        |  0.000B  |  0.000B  |
| classifier.3 |        Linear        |  1.024M  |  1.025M  |

Total `FLOPs`=124.956M, `MACs`=62.478M, `Params`=2.543M

Model size (params + buffers): 9.75 Mb

Framework & CUDA overhead: 487.00 Mb

Total RAM usage: 496.75 Mb

Floating Point Operations on forward: 123.75 MFLOPs

Multiply-Accumulations on forward: 63.90 MMACs

Direct memory accesses on forward: 62.50 MDMAs


- `resnet50`

|    Layer     |         Type         |  MACCs   |  Params  |
|-|-|-|-|
|  conv1  |       Conv2d      | 118.014M |  9.408K  |
|   bn1   |    BatchNorm2d    |  3.211M  | 128.000B |
|   relu  |        ReLU       |  0.000B  |  0.000B  |
| maxpool |     MaxPool2d     |  0.000B  |  0.000B  |
|  layer1 |     Sequential    | 685.605M | 215.808K |
|  layer2 |     Sequential    |  1.040G  |  1.220M  |
|  layer3 |     Sequential    |  1.473G  |  7.098M  |
|  layer4 |     Sequential    | 811.747M | 14.965M  |
| avgpool | AdaptiveAvgPool2d | 102.400K |  0.000B  |
|    fc   |       Linear      |  2.048M  |  2.049M  |


Total `FLOPs`=8.268G, `MACs`=4.134G, `Params`=25.557M

Model size (params + buffers): 97.70 Mb

Framework & CUDA overhead: 487.00 Mb

Total RAM usage: 584.70 Mb

Floating Point Operations on forward: 8.26 GFLOPs

Multiply-Accumulations on forward: 4.15 GMACs

Direct memory accesses on forward: 4.15 GDMAs



## Veridict

I choose `mobilenet_v3_small` model over `resnet50` because resnet50 only outperform `mobilenet_v3_small` by only 3% at the cost of nearly 10 times model size and 66 times FLOPs.

|network|f1-score-macro|MACS|FLOPS|PARAMS|
|-|-|-|-|-|
|resnet50|85.73%|4.15 GMACs|8.26 GFLOPs|25.557M|
|mobilenet_v3_small|82.06%|63.90 MMACs|123.75 MFLOPs|2.543M|



## What's Next
thing i want to do, but time did not help

1. finetune `YOLOX` for clothing classification
2. finetune with `DeepFashion2` dataset
3. train a `SimCLR` on clothing
4. test the models with out-domain images

<!-- [![fingers far from each other][screenshot-2]] -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: images/screenshot.png
[dataset]: images/dataset.jpg
[1]: images/1.jpg
[2]: images/2.jpg
[3]: images/3.jpg
[4]: images/4.jpg
[5]: images/5.jpg
[6]: images/6.jpg
[7]: images/7.jpg
[8]: images/8.jpg