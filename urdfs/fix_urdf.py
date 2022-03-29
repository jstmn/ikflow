import argparse
import xml.etree.ElementTree as ET


def remove_elements(tree, element_types):
    """ Remove elements if their type is in `element_types`
    """
    print("in remove_transmission()")
    root = tree.getroot()
    rm_elements = []
    for child in root:
        if child.tag in element_types:
            rm_elements.append(child)
    for element in rm_elements:
        root.remove(element)
    return ET.ElementTree(root)


def update_mesh_paths(tree):
    """ Update the mesh filenames from package://... to meshes/...
    """
    print("in update_mesh_paths()")

    def updated_filename(filename):
        # NOTE: May need to replace 'meshes/' this for specific robot urdfs
        mesh_idx = filename.index("meshes/")
        filename =  filename[mesh_idx:]
        if filename.endswith("DAE"):
            updated = filename[0:-3] + "STL" 
            print(f"Converting mesh filename from '{filename}' to '{updated}'")
            return updated

    root = tree.getroot()

    # Note: `materials/` not copied over for pr2_description 
    for mesh_element in root.iter("mesh"):
        mesh_filename = mesh_element.get("filename")
        updated = updated_filename(mesh_filename)
        mesh_element.set("filename", updated)
    return tree


def update_joint_types(tree, prismatic_joints, continuous_joints, revolute_joints):
    """ Set all joints to fixed. Then set all joints in `prismatic_joints` to prismatic, all joints in 
    `continuous_joints` to continuous, all joints in `revolute_joints` to revolute. 
    
    Rant about using 'etc' here:
        If there were a forth joint type, the sentence would benefit from using 'etc.'. With only three, i'm stuck 
        between a rock and a hardplace: Writing out the rule for all three joint types is overly verbose, but if I 
        don't, i'll be using 'etc' for only one value. If I only explain one joint type, the pattern may not be obvious 
        to a reader. 
    """

    print("in update_joint_types()")

    if len(prismatic_joints) == 0 and len(continuous_joints) == 0 and len(revolute_joints) == 0:
        return tree

    root = tree.getroot()

    # Set all to fixed
    for joint_element in root.iter("joint"):
        joint_element.set("type", "fixed")

    # Set active joints
    for joint_element in root.iter("joint"):
        name = joint_element.get("name")
        if name in prismatic_joints:
            joint_element.set("type", "prismatic")
        elif name in continuous_joints:
            joint_element.set("type", "continuous")
        elif name in revolute_joints:
            joint_element.set("type", "revolute")
    return tree
    

def remove_mimic(tree):
    """ Remove <mimic> tags
    """
    print("in remove_mimic()")
    root = tree.getroot()
    # Set all to fixed
    for joint_element in root.iter("joint"):

        for child in joint_element:
            if child.tag == "mimic":
                joint_element.remove(child)
                break
    return tree
    
    
""" Example usage

# Pr2
python3 urdfs/fix_urdf.py \
    --filepath=urdfs/pr2/pr2_ros.urdf \
    --output_filepath=urdfs/pr2/pr2_out.urdf \
    --prismatic_joints torso_lift_joint \
    --continuous_joints l_forearm_roll_joint l_wrist_roll_joint \
    --revolute_joints l_shoulder_pan_joint l_shoulder_lift_joint l_upper_arm_roll_joint l_elbow_flex_joint l_wrist_flex_joint \
    --remove_transmission \
    --remove_gazebo

# Baxter
python3 urdfs/fix_urdf.py \
    --filepath=urdfs/baxter/baxter_ros.urdf \
    --revolute_joints left_s0 left_s1 left_e0 left_e1 left_w0 left_w1 left_w2 \
    --output_filepath=urdfs/baxter/baxter_out.urdf \
    --remove_transmission \
    --remove_gazebo

_____
Notes

1. if `prismatic_joints`, `continuous_joints`, or `revolute_joints` is passed, all joints will be converted to fixed, 
    and then the given joint names will be updated
2. This function will remove all <mimic> elements that are children of <joint> elements. <mimic> tags mess up klampts 
    configTo/FromDriver() functions.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True, type=str)
    parser.add_argument("--output_filepath", required=True, type=str)
    parser.add_argument("--prismatic_joints", nargs="+", default=[])
    parser.add_argument("--continuous_joints", nargs="+", default=[])
    parser.add_argument("--revolute_joints", nargs="+", default=[])
    parser.add_argument("--remove_transmission", action="store_true")
    parser.add_argument("--remove_gazebo", action="store_true")

    args = parser.parse_args()

    tree = ET.parse(args.filepath)

    rm_element_types = []
    if args.remove_transmission:
        rm_element_types.append("transmission")
    if args.remove_gazebo:
        rm_element_types.append("gazebo")

    tree = remove_elements(tree, rm_element_types)
    tree = update_mesh_paths(tree)
    tree = update_joint_types(tree, args.prismatic_joints, args.continuous_joints, args.revolute_joints)
    tree = remove_mimic(tree)
    tree.write(args.output_filepath)