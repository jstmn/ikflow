

urdf source: https://github.com/gkjohnson/nasa-urdf-robots



# nasa-urdf-robots
Pre-built URDF files from the open source Robonaut 2 and Valkyrie projects from JSC for easy use without needing to install the ROS platform. The derivative URDF model files provided in this repo are covered under the same license as their original sources.

[View the models!](https://gkjohnson.github.io/nasa-urdf-robots/example/index.bundle.html)

![NASA URDF Robots](./docs/banner.png)

## Source Repositories

[nasa-jsc-robotics/val_description](https://gitlab.com/nasa-jsc-robotics/val_description/tree/77f19dc07700c30ea8f34a572600c2db0355dacc)

[nasa-jsc-robotics/r2_description](https://gitlab.com/nasa-jsc-robotics/r2_description/tree/654f4f89ff8e802cb7f80c617f0d6dd04483f4b2)

## Build Instructions

### r2_description

1. Download the repository
1. Create a `urdf` directory in the root of the repo
1. Run `bash ./convert-xacro.sh ./robots ./urdf`
1. URDF files are in `./urdf`!

### val_description

1. Initialize a catkin workspace by running
    - `mkdir -p ./catkin_ws/src`
    - `cd catkin_ws`
    - `catkin_make`
1. Download the `val_description` repository
1. Unpack the contents into `catkin_ws/src/val_description`
1. Comment out the following lines in the `catkin_ws/src/val_description/CMakeLists.txt` file
    - `catkin_python_setup()`
    - `catkin_add_nosetests(test)`
1. Run `catkin_make install`
1. URDF files are in `catkin_ws/devel/share/val_description/model/urdf`


## Modifications

- Converted R2 and Valkyrie `xacro` files to `urdf` using ROS Indigo.

## License

[NASA-1.3](https://opensource.org/licenses/NASA-1.3)
