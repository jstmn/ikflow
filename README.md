# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)


## Setup - Ubuntu

The only supported OS is Ubuntu. Everything should work in theory on Mac and Windows, I just haven't tried it out. For 
Ubuntu, there are different system wide dependencies for `Ubuntu > 21` and `Ubuntu < 21`. For example, `qt5-default` is 
not in the apt repository for Ubuntu 21.0+ so can't be installed. See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04.

### 1. Install version specific dependencies
<ins>Ubuntu >= 21.04</ins>
```
sudo apt-get install -y python3-pip qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libosmesa6
export PYOPENGL_PLATFORM=osmesa # this needs to be run every time you run a visualization script in a new terminal - annoying, I know
```

<ins>Ubuntu <= 20.x.y</ins>

(This includes 20.04 LTS, 18.04 LTS, ...)
```
sudo apt-get install -y qt5-default
```

### 2. Install common dependencies
These installation steps are the same regardless of ubuntu version
```
sudo apt-get install -y build-essential qtcreator python3.8-dev python3.8-venv git-lfs python3-wheel
python3.8 -m pip install --user virtualenv
```
Create a virtual environment:
```
python3.8 -m venv venv && source venv/bin/activate
pip install -e .
```


## Getting started

**> Example 1: Use IKFlow to generate IK solutions for the Franka Panda arm**

Evaluate the pretrained IKFlow model for the Franka Panda arm. Note that this was the same model whose performance was presented in the IKFlow paper. Note that the value for `model_name` - in this case `panda_tpm` should match an entry in `model_descriptions.yaml` 
```
python evaluate.py --testset_size=500 --model_name=panda_tpm
```

**> Example 2: Visualize the solutions returned by the `panda_tpm` model**

Run the following:
```
python visualize.py --model_name=panda_tpm --demo_name=oscillate_target
```
![alt text](../media/panda_tpm_oscillate_x-2022-08-26.gif?raw=true)

Run an interactive notebook: `jupyter notebook notebooks/robot_visualizations.ipynb`

**> Example 3: Run IKFlow yourself**

Example code for how to run IKFlow is provided in `examples/example.py`. A sample excerpt:
```
target_pose = np.array([0.5, 0.5, 0.5, 1, 0, 0, 0])
number_of_solutions = 3
solution, solution_runtime = ik_solver.solve(target_pose, number_of_solutions, refine_solutions=False)
```


## Notes
This project uses the `w,x,y,z` format for quaternions. That is all.


## Training new models

The training code uses [Pytorch Lightning](https://www.pytorchlightning.ai/) to setup and perform the training and [Weights and Biases](https://wandb.ai/) ('wandb') to track training runs and experiments. WandB isn't required for training but it's what this project is designed around. Changing the code to use Tensorboard should be straightforward (so feel free to put in a pull request for this if you want it :)).

First, create a dataset for the robot:
```
python build_dataset.py --robot_name=panda --training_set_size=2500000
```

Then start a training run:
```
# Login to wandb account - Only needs to be run once
wandb login

# Set wandb project name and entity
export WANDB_PROJECT=ikflow 
export WANDB_ENTITY=<your wandb entity name>

python train.py \
    --robot_name=panda \
    --nb_nodes=8 \
    --dim_latent_space=7 \
    --batch_size=128 \
    --learning_rate=0.0001 \
    --log_every=5000 \
    --eval_every=10000 \
    --val_set_size=500 \
    --run_description="baseline"
```

**> Add a trained model to the repo**
1. Train a model `python train.py ...`. Note down the wandb run id (it should look like '1zkh9zfo')
2. Download model with `python scripts/download_model_from_wandb_checkpoint.py --wandb_run_id=<run_id>`
3. Add the model to git lfs `git lfs track trained_models/*.pkl`
4. Add an entry for the model to 'model_descriptions.yaml' using a new alias `<new_alias>`
5. Use the model `python evaluate.py --model_name=<new_alias>`



## Common errors

1. GLUT font retrieval function when running a visualizer. Run `export PYOPENGL_PLATFORM=osmesa` and then try again. See https://bytemeta.vip/repo/MPI-IS/mesh/issues/66

```
Traceback (most recent call last):
  File "visualize.py", line 4, in <module>
    from ikflow.visualizations import _3dDemo
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

2. If you get this error: `tkinter.TclError: no display name and no $DISPLAY environment variable`, add the lines below to the top of `ik_solvers.py` (anywhere before `import matplotlib.pyplot as plt` should work).
``` python
import matplotlib
matplotlib.use("Agg")
```

## TODO
1. [ ] ~~Add CPU versions of pretrained model~~
2. [ ] Add 'light' pretrained models. These are smaller networks for faster inference
3. [ ] ~~Include a batched jacobian optimizer to enable parallelized solution refinement.~~ Note: This is in the works in a seperate repository


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