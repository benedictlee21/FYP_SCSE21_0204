# **TreeGAN**

>This repository **TreeGAN** is for _**3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions**_ paper accepted on ICCV 2019
___

## [ Paper ]
[_3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions_](https://arxiv.org/abs/1905.06292)  
(Dong Wook Shu*, Sung Woo Park*, Junseok Kwon)
___

## [Network]
TreeGAN network consists of "TreeGCN Generator" and "Discriminator".

For more details, refer our paper.
___

## [Results]
- Multi Class Generation.  
![Multi-class](https://github.com/seowok/TreeGAN/blob/master/results/fig_teaser.PNG "Motorbike, Laptop, Sofa, Guitar, Skateboard, Knife, Table, Pistol, and Car from top-left to bottom-right")

- Single Class Generation.  
![Single-class](https://github.com/seowok/TreeGAN/blob/master/results/fig_results.PNG "Plane and Chair")  

- Single Class Interpolation.  
![Single-class Interpolation](https://github.com/seowok/TreeGAN/blob/master/results/plane_interpolation.gif) 
___

## [Frechet Pointcloud Distance]
- This FPD version is used pretrained [PointNet](https://arxiv.org/abs/1612.00593).

- This FPD version is for [ShapeNet-Benchmark dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) from [_A Scalable Active Framework 
for Region Annotation in 3D Shape Collections_](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

- We also trained our model using same dataset for evaluation.

- Our **pretrained PointNet-FPD version** use only subset of official ShapeNet dataset to get [PointNet classification performance](https://github.com/fxia22/pointnet.pytorch#classification-performance) higher than 95%.

- We recommend to compose pointclouds sampled uniformly from those of ShapeNet-Benchmark dataset for training. 

- We evaluate FPD scores using 5000 samples obtained from fixed trained model with best performances.

- FPD evaluations have to use pre_statistics file for each class or all class version.

- We just provide [intermediate pretrained checkpoints and generated samples](https://drive.google.com/file/d/1FQgfBJ-tWQPE8HkqbIe9s7Kv87GfRP-z/view?usp=sharing) having fine scores when they are trained in about 1000 epochs. 
___

## [Citing]
```
@InProceedings{Shu_2019_ICCV,
               author = {Shu, Dong Wook and Park, Sung Woo and Kwon, Junseok},
               title = {3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions},
               booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
               month = {October},
               year = {2019}}
```
                           
           
## [Setting]
This project was tested on **Windows 10** / **Ubuntu 16.04**
Using _conda install_ command is recommended to setting.
### Packages
- Python 3.6
- Numpy
- Pytorch 1.0
- visdom
- Scipy 1.2.1
- Pillow
___

## [Arguments]
In our project, **arguments.py** file has almost every parameters to specify for training.

For example, if you want to train, it needs to specify _dataset_path_ argument.

### TODO
* https://github.com/wutong16/MultiModal-3DShape-Completion/blob/main/ShapeInversion/trainer.py#L131就是这里的操作，在我的 torch1.2 里其实会导致 self.model.z 直接变成了新的 z tensor，以前是 variable，它就不 update 了，改成 self.model.z.data = z.data 就好啦~
