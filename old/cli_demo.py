import argparse

from src.robot_models import get_robot
from src.model import IkflowModelParameters, IkflowModel
from src.demos.calculations import calculate_mmd
from src.demos.plots import runtime_benchmark
from src.demos.visualizations import _3dDemo
from src.utils import get_model_wrapper


""" Example usage


python3 cli_demo.py --demo_3d
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")
    parser.add_argument("--run_name", type=str, help="Run name. For example: 'blooming-forest-8582'")
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Download model from wandb regardless of whether it's already saved on disk",
    )
    parser.add_argument("--runtime_benchmark", action="store_true")
    parser.add_argument("--calculate_mmd", action="store_true")
    parser.add_argument("--demo_3d", action="store_true")
    args = parser.parse_args()

    # Build ModelWrapper and set weights
    if args.run_name is not None:
        model_wrapper, _ = get_model_wrapper(args.run_name)
    else:
        robot = get_robot("robosimian")
        base_hparams = IkflowModelParameters()
        model_wrapper = IkflowModel(base_hparams, robot)

    if args.demo_3d:
        demo_api = _3dDemo(model_wrapper)
        # demo_api.random_target_pose()
        # demo_api.oscillating_target_pose()
        # demo_api.fixed_endpose_oscilate_latent()
        demo_api.visualize_fk(solver="batch_fk")

    if args.calculate_mmd:
        calculate_mmd(model_wrapper)

    if args.runtime_benchmark:
        runtime_benchmark(model_wrapper)

    print("done")
