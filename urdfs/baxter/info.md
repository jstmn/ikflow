


Installation procedure

1. Install ros

2. Build the .urdf. Ros is required for this (see http://wiki.ros.org/pr2_description)

```
cd <nn_ik directory>/models/baxter
rosrun xacro xacro `rospack find baxter_description`/urdf/baxter.urdf.xacro > baxter_ros.urdf
cp -r /opt/ros/noetic/share/pr2_description/meshes .
cd ../../
```


```
# Format urdf
ROBOT=baxter
python3 models/fix_urdf.py \
    --filepath=models/$ROBOT/baxter_ros.urdf \
    --output_filepath=models/$ROBOT/${ROBOT}_out.urdf \
    --remove_transmission \
    --remove_gazebo

urdf_to_graphiz models/$ROBOT/${ROBOT}_out.urdf
rm ${ROBOT}.gv
mv ${ROBOT}.pdf models/$ROBOT/
```


```
cd <nn_ik directory>/models
./urdf-viz/baxter/baxter_out.urdf
```
