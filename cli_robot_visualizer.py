from src import robot_models
from src.robot_models import get_robot
from src.training_parameters import cINN_SoftflowParameters
from src.model import ModelWrapper_cINN_Softflow
from src.demos import _3dDemo


""" Example usage

python3 cli_robot_visualizer.py

"""


if __name__ == "__main__":

    """
    Borked
        Atlas,               : Visual elements are rotated
        AtlasArm,
        Kuka11dof,           : One of the joints is enormous
        Ur5,                 : Only 2 cubes are shown
        Valkyrie,            : Visual elements are rotated
        ValkyrieArm,
        ValkyrieArmShoulder,

    Working
        Robonaut2,
        Robonaut2Arm,
        PandaArm,
        Robosimian,
    """

    # robot_model = robot_models.Valkyrie()
    # robot_model = robot_models.Robonaut2()
    # robot_model = robot_models.Robonaut2Arm()
    # robot_model = robot_models.PandaArm()
    # robot_model = robot_models.Robosimian()
    # robot_model = robot_models.Kuka11dof()
    # robot_model = robot_models.Valkyrie()
    # robot_model = robot_models.ValkyrieArm()
    # robot_model = robot_models.Pr2()
    robot_model = robot_models.Baxter()

    hyper_parameters = cINN_SoftflowParameters()
    model_wrapper = ModelWrapper_cINN_Softflow(hyper_parameters, robot_model)

    demo = _3dDemo(model_wrapper)
    demo.oscillate_joints()
