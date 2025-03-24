import argparse

from ikflow.visualizations import oscillate_joints
from jrl.robots import get_robot

""" Example usage

uv run python scripts/visualize_robot.py --robot_name=panda
uv run python scripts/visualize_robot.py --robot_name=fetch
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str, help="Example: 'panda', 'baxter', ...")
    args = parser.parse_args()
    robot = get_robot(args.robot_name)
    oscillate_joints(robot)
