# TwoCnn
## Introduction
This is a reproduction of *Learning and Transferring Deep Joint Spectral-Spatial Features for Hyperspectral Classification*.

![img](img/TwoCnn.JPG)
## Requirements
* pytorch 1.3
* scikit-learn
* scipy
* visdom
## Experiment
The model was tested on three benchmark data sets: PaviaU, Salinas and KSC. The experiment is divided into three groups, with the sample size of each category being 10, the sample size of each category being 50, and the sample size of each category being 100. In order to reduce errors, each set of experiments was conducted 10 times, and the final accuracy was taken as the average of 10 experiments. The source domain and target domain pairs are respectively：Pavia-PaviaU, Indian pines-Salinas, Indian pines-KSC。

In The accuracy (%) on the PaviaU data set is shown in the following table:

<table>
<tr align="center">
<td colspan="6">PaviaU</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>78.61</td>
<td>1.88</td>
<td>90.69</td>
<td>0.71</td>
<td>94.76</td>
<td>0.45</td>
</tr>
</table>

The learning curve is as follows：

![img](img/PaviaU_sample_per_class_10_twoCnn.svg)
![img](img/PaviaU_sample_per_class_50_twoCnn.svg)
![img](img/PaviaU_sample_per_class_100_twoCnn.svg)

The accuracy (%) on the Salinas data set is shown in the table below：

<table>
<tr align="center">
<td colspan="6">Salinas</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>77.54</td>
<td>1.24</td>
<td>87.01</td>
<td>0.97</td>
<td>90.25</td>
<td>0.45</td>
</tr>
</table>

The learning curve is as follows：

![img](img/Salinas_sample_per_class_10_twoCnn.svg)
![img](img/Salinas_sample_per_class_50_twoCnn.svg)
![img](img/Salinas_sample_per_class_100_twoCnn.svg)

The accuracy (%) on the KSC data set is shown in the following table:

<table>
<tr align="center">
<td colspan="6">KSC</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>82.29</td>
<td>1.48</td>
<td>96.61</td>
<td>0.67</td>
<td>99.15</td>
<td>0.34</td>
</tr>
</table>

The learning curve is as follows：

![img](img/KSC_sample_per_class_10_twoCnn.svg)
![img](img/KSC_sample_per_class_50_twoCnn.svg)
![img](img/KSC_sample_per_class_100_twoCnn.svg)
## Runing the code
Pre-trained model `python train.py --name xx --epoch xx --lr xx`

Fine-tuning `python CrossTrain.py --name xx --epoch xx --lr xx` 
