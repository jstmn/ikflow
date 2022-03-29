# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper "IKFlow: Generating Diverse Inverse Kinematics Solutions"

### Setup

1. If python3.8 isn't isntalled and `sudo` isn't available:

Follow instructions here: https://stackoverflow.com/a/61890236/5191069

```
# If on ubuntu
sudo apt-get install python3-dev qt5-default

# Create and activate a virtual environment 
python3.8 -m pip install --user virtualenv
python3.8 -m venv venv
source venv/bin/activate
pip install wheel
python -m pip install -r requirements.txt

# Install git submodules
git submodule init; git submodule update

# Install FRiEA
cd src/FrEIA; python setup.py develop

# (optional) Add venv to jupyter notebook
pip install ipykernel; python -m ipykernel install --user --name=venv

# 
```