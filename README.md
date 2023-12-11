# AGIMUS 2023 Winter School

[Main website](https://aws.sciencesconf.org/)

[Chat room](https://matrix.to/#/#aws-main-hall:laas.fr)

## Tutorials

In this winter school, we will cover three main different topics:

- [Simulation](./simulation)
    - [Simulation #1: Rigid body dynamics](./simulation/sim1_rigid_body/), [slides](./simulation/slides/sim1.pdf)
- [Optimal control](./ocp/)
- [Motion planning]

## Installing dependencies

Dependencies for the coursework include specific versions of Pinocchio, hpp-fcl, crocoddyl, alligator, and other software that have yet to be fully released.
We provide two ways to install the required packages for Mac OS and Linux: conda/mamba or docker.

### Conda installation [Mac OS Intel, Mac OS ARM, Linux x64]

All the required packages are available on the following [channel](https://anaconda.org/agm-ws-2023/repo).
Conda can be easily installed on your machine by following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can install a package by typing in your terminal:
```bash
conda create -n aws python=3.10
conda activate aws
conda install -c agm-ws-2023 my_package_name
```
The two first lines create a new environment named `aws` and then activate it.
The third line installs `my_package_name`using the [channel](https://anaconda.org/agm-ws-2023) of the AGIMUS winter school where the packages have been compiled.

You can also consider installing additional tools via pip, like:
```bash
pip install tqdm meshcat ipython
```

We also invite you to leverage [visual studio code](https://code.visualstudio.com/) to play with the Jupyter notebooks.
Don't forget to install the [Jupyter module](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for [visual studio code](https://code.visualstudio.com/).

### Docker installation
[[Add Docker instructions here]]
