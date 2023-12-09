# AGIMUS 2023 Winter School: Perception

The perception course will cover two main areas: (i) object 6D pose estimation from images and (ii) object tracking from videos.
The former is based on [HappyPose](https://github.com/agimus-project/happypose), our open source reimplementation of state-of-the-art object pose estimation methods called CosyPose and MegaPose.
The latter is based on DLR Tracker... **Mederic: TBD**

The course is based on following publications:

- Fourmy, M., Priban, V., Behrens, J. K., Mansard, N., Sivic, J., & Petrik, V. (2023). **Visually Guided Model Predictive Robot Control via 6D Object Pose Localization and Tracking**. In review for ICRA 2023. [ArXiv](https://arxiv.org/pdf/2311.05344), [project page](https://data.ciirc.cvut.cz/public/projects/2023VisualMPC/).
- Labbé, Y., Manuelli, L., Mousavian, A., Tyree, S., Birchfield, S., Tremblay, J., Carpentier, J., Aubry, M., Fox, D. and Sivic, J., (2022). **Megapose: 6d pose estimation of novel objects via render & compare**. [ArXiv](https://arxiv.org/abs/2212.06870), [project page](https://megapose6d.github.io/).
- Labbé, Y., Carpentier, J., Aubry, M., & Sivic, J. (2020). **Cosypose: Consistent multi-view multi-object 6d pose estimation**. In ECCV 2020. [ArXiv](https://arxiv.org/abs/2008.08465), [project page](https://www.di.ens.fr/willow/research/cosypose/).



## Outline

- Course on perception and tracking
  - 45 mins - Presentation on 6D object pose estimation from images [V. Petrik]
  - 45 mins - Presentation on 6D object pose tracking [M. Fourmy]
- Coding tutorial [V. Petrik, M. Fourmy, K. Zorina, M. Cifka]
  - Object detection in image
  - Object pose estimation for objects known at train time (CosyPose)
  - Object pose estimation for objects unknown at train time (MegaPose)
  - Object tracking
  - Object pose estimation and tracking pipeline

## Installation

To make code easily replicable on laptops, we will install CPU version only. The CPU version is however slow, see [HappyPose](https://github.com/agimus-project/happypose) for installation of GPU version for you future projects.

```bash
# Create Conda environment
conda create -n aws_perception python=3.9
conda activate aws_perception

# HappyPose installation
pip install "git+https://github.com/agimus-project/happypose.git#egg=happypose[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### Downloading HappyPose data

Object pose estimators are based on pre-trained networks and the dataset of objects.
To be able to create/run the tutorial code you need to download both with:
```
TBD
```

### Downloading Tutorial data

For the tutorial, we pre-recorded the sequences of images that we will analyze.
Please, download the data into this repository (folder `winter-school-2023/perception/data`) from:
```TBD```


## Tutorial

The tutorial is split into several scripts. In all scripts there are places marked with `TODO` that you need to complete in order for script to work.


### Object Detection

`TBD show input - expected output`

### Object Pose Estimation (CosyPose)

`TBD show input - expected output`

### Object Pose Estimation (MegaPose)

`TBD show input - expected output`

### Object Tracking

`TBD show input - expected output`

### Estimate and Track pipeline

The goal for you is to complete this pipeline and run it either on our pre-recorded sequences or on your camera.
For camera, we 3D printed the cup object from YCBV dataset, if you want to replicate the result at home, use this object for 3D printing: `TBD` and use red PLA filament for printing.

`TBD show input - expected output`

## Contact

In case of any question do not hesitate to contact us:
- Vladimir Petrik, vladimir.petrik@cvut.cz, https://petrikvladimir.github.io/
- Mederic Fourmy, mederic.fourmy@cvut.cz, `TBD`
