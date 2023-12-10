# Contact dynamics: an introduction

This introductory lecture follows these two following reference papers:
  - Carpentier, J., Budhiraja, R., & Mansard, N. (2021, July). [Proximal and sparse resolution of constrained dynamic equations](https://inria.hal.science/hal-03271811/file/rss-proximal-and-sparse.pdf). In Robotics: Science and Systems 2021.
  - Le Lidec, Q., Jallet, W., Montaut, L., Laptev, I., Schmid, C., & Carpentier, J. (2023). [Contact Models in Robotics: a Comparative Analysis](https://hal.science/hal-04067291v1/preview/lelidec2023contacts.pdf). 

## Content

* Introduction to the main concepts of contact simulation
* Numerics of simulation
* Robotics simulators

## Tutorial

* tp_sim3.py : contact simulation

### Requirements

You can go ahead and install all the required packages using conda. 
Conda can be easily installed on your machine by following this [install procedure](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
After the installation of conda, you can proceed to the installation of the required packages:
 
```bash
conda create -n aws-sim3 python=3.10
conda activate aws-sim3
conda install -c agm-ws-2023 pinocchio exemple-robot-data
pip install tqdm meshcat ipython
```
