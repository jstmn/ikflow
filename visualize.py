import argparse
import yaml

from utils.visualizations import _3dDemo
from utils.utils import get_ik_solver, set_seed

set_seed()

with open("model_descriptions.yaml", "r") as f:
    MODEL_DESCRIPTIONS = yaml.safe_load(f)

""" Example usage

python visualize.py --model_name=panda_tpm

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--model_name", type=str, help="Name of the saved model to look for in model_descriptions.yaml")
    parser.add_argument("--visualization_program", type=str, help="One of [---TODO---]")
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

    model_weights_filepath = MODEL_DESCRIPTIONS[args.model_name]["model_weights_filepath"]
    robot_name = MODEL_DESCRIPTIONS[args.model_name]["robot_name"]
    hparams = MODEL_DESCRIPTIONS[args.model_name]

    # Build GenerativeIKSolver and set weights
    ik_solver, hyper_parameters = get_ik_solver(model_weights_filepath, robot_name, hparams)

    demo = _3dDemo(ik_solver)
    # demo.oscillate_joints()
    demo.oscillating_target_pose()


