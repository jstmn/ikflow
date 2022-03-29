


Installation procedure

1. Install ros

2. Build the .urdf. Ros is required for this (see http://wiki.ros.org/pr2_description)

```
cd <nn_ik directory>/models/pr2
rosrun xacro xacro `rospack find pr2_description`/robots/pr2.urdf.xacro > pr2_ros.urdf
cp -r /opt/ros/noetic/share/pr2_description/meshes .
cd ../../
```

Convert .dae to .obj
```
ctmconv 

```



```
# Format urdf
ROBOT=pr2
python3 models/fix_urdf.py \
    --filepath=models/$ROBOT/pr2_ros.urdf \
    --output_filepath=models/$ROBOT/${ROBOT}_out.urdf \
    --remove_transmission \
    --remove_gazebo

urdf_to_graphiz models/$ROBOT/${ROBOT}_out.urdf
rm ${ROBOT}.gv
mv ${ROBOT}.pdf models/$ROBOT/
```


Install `urdf-viz` (This isn't working, I think b/c of the .dae files. Try again once .dae's are converted to .obj/.stls)
```
sudo apt-get install cmake xorg-dev libglu1-mesa-dev
# Go to https://github.com/openrr/urdf-viz/releases, download the linux binary to '<nn_ik directory>/models'
cd <nn_ik directory>/models
chmod +x urdf-viz
./urdf-viz/pr2/pr2_out.urdf
```
