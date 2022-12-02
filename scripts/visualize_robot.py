import argparse

# Update sys.path so you can run this file from the projects root directory
import sys
import os


from ikflow.visualizations import oscillate_joints
from jkinpylib.robots import get_robot

""" Example usage

python scripts/visualize_robot.py --robot_name=panda_arm
python scripts/visualize_robot.py --robot_name=robonaut2
python scripts/visualize_robot.py --robot_name=valkyrie
python scripts/visualize_robot.py --robot_name=baxter
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str, help="Example: 'panda_arm', 'baxter', ...")
    args = parser.parse_args()

    """
        Atlas,
        Valkyrie,            
        Robonaut2,
    
    Borked
        ...

    Working
        PandaArm
    """

    robot = get_robot(args.robot_name)
    oscillate_joints(robot)
