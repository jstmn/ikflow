# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)


## Setup - Ubuntu

TODO: Add instructions for running on Mac

**Install python3.8 (if on ubuntu 22+**
```
sudo apt update && sudo apt upgrade
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8 -y
```

**Install base ubuntu dependencies**

Note: `qt5-default` is not in the apt repository for Ubuntu 21.0+ and thus cannot be installed. 
See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04. Therefor, if running Ubuntu `> 21.0`:
```
sudo apt-get install -y build-essential qtcreator python3.8-dev python3-pip python3.8-venv git-lfs 
sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libosmesa6
export PYOPENGL_PLATFORM=osmesa # this needs to be run every time you run a visualization script in a new terminal - annoying, I know
git lfs pull origin master # Download pretrained models
```
Otherwise if running Ubuntu `< 21.0+`:
```
sudo apt-get install build-essential qtcreator qt5-default python3.8-dev python3.8-pip python3.8-venv git-lfs
git lfs pull origin master # Download pretrained models
```

**Create a virtual environment**
```
python3.8 -m pip install --user virtualenv
python3.8 -m venv venv
source venv/bin/activate
pip install wheel
python -m pip install -r requirements.txt

# Install git submodules
git submodule init; git submodule update

# Install thirdparty libraries
cd thirdparty/FrEIA && python setup.py develop && cd ../../
```

**Build datasets**

This will build a dataset for the Frank Panda arm.  
```
python build_dataset.py --robot_name=panda_arm --training_set_size=2500000
```

## Examples

Evaluate the pretrained IKFlow model for the Franka Panda arm. Note that this was the same model whose performance was presented in the IKFlow paper.
```
python evaluate.py \
    --samples_per_pose=50 \
    --testset_size=25 \
    --model_name=panda_tpm
```

Visualize the solutions returned by the `panda_tpm` model:
```
python visualize.py --model_name=panda_tpm
```
![alt text](../media/panda_tpm_oscillate_x-2022-08-26.gif?raw=true)

## Common errors

1. Pickle error when loading a pretrained model from gitlfs. This happens when the pretrained models haven't been downloaded completely by gitlfs. For example, they may be 134 bytes (they should be 100+ mb). Run `git lfs pull origin master` 
```
Traceback (most recent call last):
  File "evaluate.py", line 117, in <module>
    ik_solver, hyper_parameters = get_ik_solver(model_weights_filepath, robot_name, hparams)
  File "/home/jstm/Projects/ikflow/utils/utils.py", line 55, in get_ik_solver
    ik_solver.load_state_dict(model_weights_filepath)
  File "/home/jstm/Projects/ikflow/utils/ik_solvers.py", line 278, in load_state_dict
    self.nn_model.load_state_dict(pickle.load(f))
_pickle.UnpicklingError: invalid load key, 'v'.
```

2. GLUT font retrieval function when running a visualizer. Run `export PYOPENGL_PLATFORM=osmesa` and then try again. See https://bytemeta.vip/repo/MPI-IS/mesh/issues/66

```
Traceback (most recent call last):
  File "visualize.py", line 4, in <module>
    from utils.visualizations import _3dDemo
  File "/home/jstm/Projects/ikflow/utils/visualizations.py", line 10, in <module>
    from klampt import vis
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/__init__.py", line 3, in <module>
    from .glprogram import *
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/glprogram.py", line 11, in <module>
    from .glviewport import GLViewport
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/glviewport.py", line 8, in <module>
    from . import gldraw
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/gldraw.py", line 10, in <module>
    from OpenGL import GLUT
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/GLUT/__init__.py", line 5, in <module>
    from OpenGL.GLUT.fonts import *
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/GLUT/fonts.py", line 20, in <module>
    p = platform.getGLUTFontPointer( name )
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/platform/baseplatform.py", line 350, in getGLUTFontPointer
    raise NotImplementedError( 
NotImplementedError: Platform does not define a GLUT font retrieval function
```

## TODO
1. [ ] Add CPU versions of pretrained model
2. [ ] Add 'light' pretrained models. These are smaller networks for faster inference
3. [ ] Include a batched jacobian optimizer to enable parallelized solution refinement   



## Citation
```
@ARTICLE{9793576,
  author={Ames, Barrett and Morgan, Jeremy and Konidaris, George},
  journal={IEEE Robotics and Automation Letters}, 
  title={IKFlow: Generating Diverse Inverse Kinematics Solutions}, 
  year={2022},
  volume={7},
  number={3},
  pages={7177-7184},
  doi={10.1109/LRA.2022.3181374}
}
```