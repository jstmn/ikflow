from typing import List, Dict
from time import time

from src.supporting_types import ModelAccuracyResults
from src.math_utils import rotation_matrix_from_quaternion, geodesic_distance
from src import config
from src.dataset import Dataset
from src.model import ModelWrapper, ModelWrapper_MultiCINN

from tqdm import tqdm
import numpy as np
import torch


# TODO(@jeremysm): Change this to tage a numpy array as a testset, instead of a Dataset
def test_model_accuracy(
    model_wrapper: ModelWrapper, dataset: Dataset, n_target_poses: int, samples_per_pose: int
) -> ModelAccuracyResults:
    """
    Evaluate the nn_model on n_poses_per_evaluation end points, querrying samples_per_pose samples from the nn_model for each endpoint
    """

    if model_wrapper.nn_model_type == ModelWrapper_MultiCINN.nn_model_type:
        for model in model_wrapper.nn_model:
            model.eval()
    else:
        model_wrapper.nn_model.eval()

    l2_errs: List[List[float]] = []
    angular_errors: List[List[float]] = []
    model_runtimes = []

    _3d = model_wrapper.robot_model.dim_y == 7

    with torch.inference_mode():

        for i in range(n_target_poses):
            ee_pose_target = dataset.endpoints_te_numpy[i]
            samples, model_runtime = model_wrapper.make_samples(ee_pose_target, samples_per_pose)
            ee_pose_ikflow = model_wrapper.robot_model.forward_kinematics(
                samples[:, 0 : model_wrapper.robot_model.dim_x].cpu().detach().numpy()
            )
            model_runtimes.append(model_runtime)

            if _3d:

                # Positional Error
                pos_l2errs = np.linalg.norm(ee_pose_ikflow[:, 0:3] - ee_pose_target[0:3], axis=1)

                # Angular Error
                rot_target = np.tile(ee_pose_target[3:], (samples_per_pose, 1))
                rot_output = ee_pose_ikflow[:, 3:]

                output_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_output).to(config.device))
                target_R9 = rotation_matrix_from_quaternion(torch.Tensor(rot_target).to(config.device))
                angular_rad_errs = geodesic_distance(target_R9, output_R9).cpu().data.numpy()

            else:
                pos_l2errs = np.linalg.norm(ee_pose_ikflow[:, 0:2] - ee_pose_target[0:2], axis=1)
                angular_rad_errs = np.zeros(pos_l2errs.shape)

            angular_errors.append(angular_rad_errs)
            l2_errs.append(pos_l2errs)

    return ModelAccuracyResults(l2_errs, angular_errors, model_runtimes)
