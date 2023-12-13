# AGIMUS 2023 Winter School

[Main website](https://aws.sciencesconf.org/)

[Chat room](https://matrix.to/#/#aws-main-hall:laas.fr)

## Tutorials

In this winter school, we will cover three main different topics:

- [Simulation](./simulation)
    - [Simulation #1: Rigid body dynamics](./simulation/sim1_rigid_body/), [slides](./simulation/slides/sim1.pdf)
    - [Simulation #2: Collision detection](./simulation/sim2_collision), [slides](./simulation/slides/sim2.pdf)
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
conda install -c agm-ws-2023 -c conda-forge my_package_name
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

First, get the image:
- `docker pull reg.saurel.me/aws` on the Eduroam network
- `docker pull reg-w.saurel.me/aws` on the wired network

To run it, if you are on Linux, PAL provides helpers at https://github.com/pal-robotics/pal_docker_utils to allow
running graphical applications.

You could also run it with eg. `docker run --rm --net=host -it reg.saurel.me/aws`, and once inside run the `docker-vnc`
command to retrieve a link like http://localhost:6080/vnc.html?host=localhost&port=6080 : clic on it to open a virtual
desktop in your webbrowser.

#### Docker use

start with a `git clone https://github.com/agimus-project/winter-school-2023`

- for HPP tutorials: it should work out of the box
- for ROS tutorials: `source /opt/pal/alum/setup.bash`
- for others, run `/opt/miniconda3/bin/conda init && bash`, then
    - for Pinocchio / Crocoddyl tutorials: `conda activate /aws1`
        - for the mim-solvers part, you'll need:
            `sudo /opt/miniconda3/bin/conda install -yp /aws1 agm-ws-2023-2::mim-solvers`
    - for Happypose tutorials: `conda activate /aws2`
