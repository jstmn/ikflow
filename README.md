# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper "IKFlow: Generating Diverse Inverse Kinematics Solutions"

### Setup - Ubuntu

TODO: Add instructions for running on Mac


*Install base ubuntu dependencies*
```
# Ubuntu < 21.0
sudo apt-get install build-essential qtcreator qt5-default python3.8-dev python3.8-pip python3.8-venv  
```

Note: `qt5-default` is not in the apt repository for Ubuntu 21.0+ and thus cannot be installed. See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04
```
# Ubuntu 21.0+
sudo apt-get install build-essential qtcreator python3.8-dev python3-pip python3.8-venv 
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

*Create virtual environment*
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




XXXXXXXXXXXXXXX

pip install torch numpy klampt matplotlib pytorch_lightning tqdm wandb kinpy
pip freeze > requirments.txt

git submodule add https://github.com/tonyduan/mdn.git thirdparty/mdn
git submodule add https://github.com/VLL-HD/FrEIA.git thirdparty/FrEIA

[submodule "src/mdn"]
	path = src/mdn
	url = https://github.com/tonyduan/mdn.git
[submodule "src/FrEIA"]
	path = src/FrEIA
	url = https://github.com/VLL-HD/FrEIA.git