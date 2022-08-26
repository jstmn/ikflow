# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)


## Setup - Ubuntu

TODO: Add instructions for running on Mac

**Install base ubuntu dependencies**

Note: `qt5-default` is not in the apt repository for Ubuntu 21.0+ and thus cannot be installed. 
See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04. Therefor, if running Ubuntu `> 21.0`:
```
sudo apt-get install build-essential qtcreator python3.8-dev python3-pip python3.8-venv git-lfs 
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libosmesa6
export PYOPENGL_PLATFORM=osmesa
```
Otherwise if running Ubuntu `< 21.0+`:
```
sudo apt-get install build-essential qtcreator qt5-default python3.8-dev python3.8-pip python3.8-venv git-lfs
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