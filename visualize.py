import argparse
import yaml

from src import visualizations
from src.utils import get_ik_solver, set_seed

set_seed()

with open("model_descriptions.yaml", "r") as f:
    MODEL_DESCRIPTIONS = yaml.safe_load(f)

VIS_FN_ARGUMENTS = {"oscillate_target_pose": {"nb_sols": 10, "fixed_latent_noise": True}, "oscillate_latent": {}}

""" Example usage. Note that `model_name` should match an entry in `model_descriptions.yaml`

python visualize.py --model_name=panda_tpm --demo_name=oscillate_latent
python visualize.py --model_name=panda_tpm --demo_name=oscillate_target_pose

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--model_name", type=str, help="Name of the saved model to look for in model_descriptions.yaml")
    parser.add_argument("--visualization_program", type=str, help="One of [---TODO---]")
    parser.add_argument("--demo_name", type=str, help="One of ['oscillate_latent', 'oscillate_target_pose']")
    args = parser.parse_args()

    assert args.demo_name in [x for x in dir(visualizations) if not x[0] == "_"]
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

    # Build IkflowSolver and set weights
    ik_solver, hyper_parameters = get_ik_solver(model_weights_filepath, robot_name, hparams)

    fn = getattr(visualizations, args.demo_name)
    fn(ik_solver, **VIS_FN_ARGUMENTS[args.demo_name])
