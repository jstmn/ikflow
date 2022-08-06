# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)


## Setup - Ubuntu

TODO: Add instructions for running on Mac

**Install base ubuntu dependencies**

- If running Ubuntu `< 21.0`:
```
sudo apt-get install build-essential qtcreator qt5-default python3.8-dev python3.8-pip python3.8-venv  git-lfs
```

- Else if running Ubuntu `> 21.0+`:
Note: `qt5-default` is not in the apt repository for Ubuntu 21.0+ and thus cannot be installed. See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04
```
sudo apt-get install build-essential qtcreator python3.8-dev python3-pip python3.8-venv git-lfs 
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

**Create virtual environment**
```
# Create and activate a virtual environment 
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
```
python build_dataset.py --robot_name=panda_arm --training_set_size=2500000
```

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