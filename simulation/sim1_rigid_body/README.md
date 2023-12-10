# Rigid body dynamics: an introduction

This introductory lecture follows these two following references:
  - Featherstone, R. (2014). [Rigid body dynamics algorithms](https://link.springer.com/content/pdf/10.1007/978-1-4899-7560-7.pdf). Springer
  - Featherstone, R. (2010). [A beginner's guide to 6-d vectors (part 1)](). IEEE robotics & automation magazine, 17(3), 83-94
  - Featherstone, R. (2010). [A beginner's guide to 6-D vectors (part 2) - tutoria](). IEEE robotics & automation magazine, 17(4), 88-99.
  - Carpentier, J., Saurel, G., Buondonno, G., Mirabel, J., Lamiraux, F., Stasse, O., & Mansard, N. (2019, January). [The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives](https://hal.laas.fr/hal-01866228/document). In 2019 IEEE/SICE International Symposium on System Integration (SII)

## Content

* Introduction to spatial quantities: spatial velocity, force, inertia, etc.
* Introduction to poly-articulated systems: model and data
* Forward and inverse kinematics
* Forward and inverse dynamic algorithms
* Analytical derivatives of forward and inverse dynamics

## Requirements

You can go ahead and install all the required packages using conda. 
Conda can be easily installed on your machine by following this [install procedure](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
After the installation of conda, you can proceed to the installation of the required packages:
 
```bash
conda create -n aws-sim1 python=3.10
conda activate aws-sim1
conda install -c agm-ws-2023 pinocchio exemple-robot-data
pip install tqdm meshcat ipython
```

## Tutorials

* [1_geometry_and_dynamics.ipynb](./1_geometry_and_dynamics.ipynb) : introduction to the Pinocchio framework
* [2_derivatives.ipynb](./2_derivatives.ipynb): introduction to the analytical derivatives of forward and inverse dynamics

## Advanced tutorials using Casadi

If you are already familiar with Pinocchio and its features, we invite to have fun with the support of Pinocchio and the Casadi framework supporting automatic differentiation through the computational graph of Pinocchio. Below, you will find a list of progressive tutorials to dive into this brandly new aspect of Pinocchio.

* [0_setup.ipynb](https://github.com/nmansard/jnrh2023/blob/main/0_setup.ipynb)
* [1_invgeom.ipynb](https://github.com/nmansard/jnrh2023/blob/main/1_invgeom.ipynb)
* [2_trajopt_geom.ipynb](https://github.com/nmansard/jnrh2023/blob/main/2_trajopt_geom.ipynb)
* [3_contact_dynamics.ipynb](https://github.com/nmansard/jnrh2023/blob/main/3_contact_dynamics.ipynb)
* [4_with_obstacles.ipynb](https://github.com/nmansard/jnrh2023/blob/main/4_with_obstacles.ipynb)

More details are available [here](https://github.com/nmansard/jnrh2023).
