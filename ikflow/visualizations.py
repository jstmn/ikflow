from typing import List, Callable, Any
from time import sleep
from dataclasses import dataclass

from ikflow.ikflow_solver import IkflowSolver
from ikflow import config
from ikflow.utils import get_solution_errors
from jkinpylib.robot import Robot

from klampt.model import coordinates, trajectory
from klampt import vis
from klampt import WorldModel
import numpy as np
import torch
import torch.optim


def _run_demo(
    robot_model: Robot,
    n_worlds: int,
    setup_fn: Callable[[List[WorldModel]], None],
    loop_fn: Callable[[List[WorldModel], Any], None],
    viz_update_fn: Callable[[List[WorldModel], Any], None],
    demo_state: Any = None,
    time_p_loop: float = 2.5,
    title="Anonymous demo",
    load_terrain=False,
):
    """Internal function for running a demo."""

    worlds = [robot_model._klampt_world_model.copy() for _ in range(n_worlds)]

    # TODO: Adjust terrain height for each robot
    if load_terrain:
        terrain_filepath = "urdfs/klampt_resources/terrains/plane.off"
        res = worlds[0].loadTerrain(terrain_filepath)
        assert res, f"Failed to load terrain '{terrain_filepath}'"
        vis.add("terrain", worlds[0].terrain(0))

    setup_fn(worlds)

    vis.setWindowTitle(title)
    vis.show()
    while vis.shown():
        # Modify the world here. Do not modify the internal state of any visualization items outside of the lock
        vis.lock()

        loop_fn(worlds, demo_state)
        vis.unlock()

        # Outside of the lock you can use any vis.X functions, including vis.setItemConfig() to modify the state of objects
        viz_update_fn(worlds, demo_state)
        sleep(time_p_loop)
    vis.kill()


""" Class contains functions that show generated solutions to example problems

"""


def visualize_fk(ik_solver: IkflowSolver, solver="klampt"):
    """Set the robot to a random config. Visualize the poses returned by fk"""

    assert solver in ["klampt", "batch_fk"]

    n_worlds = 1
    time_p_loop = 30
    title = "Visualize poses returned by FK"

    robot = ik_solver.robot

    def setup_fn(worlds):
        vis.add(f"robot", worlds[0].robot(0))
        vis.setColor((f"robot", robot.end_effector_link_name), 0, 1, 0, 0.7)

        # Axis
        vis.add("coordinates", coordinates.manager())
        vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
        vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

    from klampt.math import so3

    def loop_fn(worlds, _demo_state):
        x_random = robot.sample_joint_angles(1)
        q_random = robot._x_to_qs(x_random)
        worlds[0].robot(0).setConfig(q_random[0])

        if solver == "klampt":
            fk = robot.forward_kinematics_klampt(x_random)
            ee_pose = fk[0, 0:7]
            vis.add("ee", (so3.from_quaternion(ee_pose[3:]), ee_pose[0:3]), length=0.15, width=2)
        else:
            # (B x 3*(n+1) )
            x_torch = torch.from_numpy(x_random).float().to(config.device)
            fk = robot.forward_kinematics_batch(x_torch)
            ee_pose = fk[0, 0:3]
            vis.add("ee", (so3.identity(), ee_pose[0:3]), length=0.15, width=2)

    def viz_update_fn(worlds, _demo_state):
        return

    _run_demo(robot, n_worlds, setup_fn, loop_fn, viz_update_fn, time_p_loop=time_p_loop, title=title)


def oscillate_latent(ik_solver: IkflowSolver):
    """Fixed end pose, oscillate through the latent space"""

    n_worlds = 2
    time_p_loop = 0.01
    title = "Fixed end pose with oscillation through the latent space"
    robot = ik_solver.robot

    def setup_fn(worlds):
        vis.add(f"robot_1", worlds[1].robot(0))
        vis.setColor(f"robot_1", 1, 1, 1, 1)
        vis.setColor((f"robot_1", robot.end_effector_link_name), 1, 1, 1, 0.71)

        # Axis
        vis.add("coordinates", coordinates.manager())
        vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
        vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

    target_pose = np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0])
    rev_input = torch.zeros(1, ik_solver.network_width).to(config.device)

    @dataclass
    class DemoState:
        counter: int

    def loop_fn(worlds, _demo_state):
        for i in range(ik_solver.network_width):
            rev_input[0, i] = 0.25 * np.cos(_demo_state.counter / 25) - 0.1 * np.cos(_demo_state.counter / 250)

        # Get solutions to pose of random sample
        ik_solutions = ik_solver.make_samples(target_pose, 1, latent_noise=rev_input)[0]
        qs = robot._x_to_qs(ik_solutions)
        worlds[1].robot(0).setConfig(qs[0])

        # Update _demo_state
        _demo_state.counter += 1

    def viz_update_fn(worlds, _demo_state):
        return

    demo_state = DemoState(counter=0)
    _run_demo(
        robot, n_worlds, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=time_p_loop, title=title
    )


# TODO(@jeremysm): Add/flesh out plots. Consider plotting each solutions x, or error
def oscillate_target_pose(ik_solver: IkflowSolver, nb_sols=5, fixed_latent_noise=True):
    """Oscillating target pose"""

    initial_target_pose = np.array([0, 0.5, 0.25, 1.0, 0.0, 0.0, 0.0])
    time_p_loop = 0.01
    title = "Solutions for oscillating target pose"
    latent = None
    if fixed_latent_noise:
        latent = torch.randn((nb_sols, ik_solver.network_width)).to(config.device)

    robot = ik_solver.robot

    def setup_fn(worlds):
        vis.add("coordinates", coordinates.manager())
        for i in range(len(worlds)):
            vis.add(f"robot_{i}", worlds[i].robot(0))
            vis.setColor(f"robot_{i}", 1, 1, 1, 1)
            vis.setColor((f"robot_{i}", robot.end_effector_link_name), 1, 1, 1, 0.71)

        # Axis
        vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
        vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

        # Add target pose plot
        vis.addPlot("target_pose")
        vis.logPlot("target_pose", "target_pose x", 0)
        vis.setPlotDuration("target_pose", 5)
        vis.addPlot("solution_error")
        vis.addPlot("solution_error")
        vis.logPlot("solution_error", "l2 (mm)", 0)
        vis.logPlot("solution_error", "angular (deg)", 0)
        vis.setPlotDuration("solution_error", 5)
        vis.setPlotRange("solution_error", 0, 25)

    @dataclass
    class DemoState:
        counter: int
        target_pose: np.ndarray
        ave_l2_error: float
        ave_angular_error: float

    def loop_fn(worlds, _demo_state):
        # TODO(@jeremysm): Implement a method to easily change end pose arcs

        # Update target pose
        x = 0.25 * np.sin(_demo_state.counter / 50)
        _demo_state.target_pose[0] = x

        # Circle with radius .5 centered at origin
        # x = .5 * np.cos(_demo_state.counter / 25)
        # y = .5 * np.sin(_demo_state.counter / 25)
        # _demo_state.target_pose[0] = x
        # _demo_state.target_pose[1] = y

        # Get solutions to pose of random sample
        ik_solutions = ik_solver.make_samples(_demo_state.target_pose, nb_sols, latent_noise=latent)[0]
        l2_errors, ang_errors = get_solution_errors(ik_solver.robot, ik_solutions, _demo_state.target_pose)
        _demo_state.ave_l2_error = np.mean(l2_errors) * 1000
        _demo_state.ave_ang_error = np.rad2deg(np.mean(ang_errors))

        # Update viz with solutions
        qs = robot._x_to_qs(ik_solutions)
        for i in range(nb_sols):
            worlds[i].robot(0).setConfig(qs[i])

        # Update _demo_state
        _demo_state.counter += 1

    def viz_update_fn(worlds, _demo_state):
        line = trajectory.Trajectory([1, 0], [_demo_state.target_pose[0:3], [0, 0, 0]])
        vis.add("target_pose.", line)
        vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])
        vis.logPlot("solution_error", "l2 (mm)", _demo_state.ave_l2_error)
        vis.logPlot("solution_error", "angular (deg)", _demo_state.ave_ang_error)

    demo_state = DemoState(counter=0, target_pose=initial_target_pose.copy(), ave_l2_error=0, ave_angular_error=0)
    _run_demo(
        robot, nb_sols, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=time_p_loop, title=title
    )


def random_target_pose(ik_solver: IkflowSolver, nb_sols=5):
    """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""

    def setup_fn(worlds):
        vis.add(f"robot_goal", worlds[0].robot(0))
        vis.setColor(f"robot_goal", 0.5, 1, 1, 0)
        vis.setColor((f"robot_goal", ik_solver.end_effector_link_name), 0, 1, 0, 0.7)

        for i in range(1, nb_sols + 1):
            vis.add(f"robot_{i}", worlds[i].robot(0))
            vis.setColor(f"robot_{i}", 1, 1, 1, 1)
            vis.setColor((f"robot_{i}", ik_solver.end_effector_link_name), 1, 1, 1, 0.71)

    def loop_fn(worlds, _demo_state):
        # Get random sample
        random_sample = self.robot_model.sample_joint_angles(1)
        random_sample_q = self.robot_model._x_to_qs(random_sample)
        worlds[0].robot(0).setConfig(random_sample_q[0])
        target_pose = self.robot_model.forward_kinematics_klampt(random_sample)[0]

        # Get solutions to pose of random sample
        ik_solutions = self.ik_solver.make_samples(target_pose, nb_sols)[0]
        qs = self.robot_model._x_to_qs(ik_solutions)
        for i in range(nb_sols):
            worlds[i + 1].robot(0).setConfig(qs[i])

    time_p_loop = 2.5
    title = "Solutions for randomly drawn poses - Green link is the target pose"

    def viz_update_fn(worlds, _demo_state):
        return

    self._run_demo(nb_sols + 1, setup_fn, loop_fn, viz_update_fn, time_p_loop=time_p_loop, title=title)


def oscillate_joints(robot: Robot):
    """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""

    inc = 0.01

    class DemoState:
        def __init__(self):
            self.q = np.array([lim[0] for lim in robot.actuated_joints_limits])
            self.increasing = True

    def setup_fn(worlds):
        vis.add("robot", worlds[0].robot(0))
        vis.setColor("robot", 1, 0.1, 0.1, 1)
        assert len(worlds) == 1

    def loop_fn(worlds, _demo_state):
        no_change = True
        for i in range(robot.n_dofs):
            joint_limits = robot.actuated_joints_limits[i]
            if _demo_state.increasing:
                if _demo_state.q[i] < joint_limits[1]:
                    _demo_state.q[i] += inc
                    no_change = False
                    break
            else:
                if _demo_state.q[i] > joint_limits[0]:
                    _demo_state.q[i] -= inc
                    no_change = False
                    break
        if no_change:
            _demo_state.increasing = not _demo_state.increasing

        q = robot._x_to_qs(np.array([_demo_state.q]))
        worlds[0].robot(0).setConfig(q[0])

    time_p_loop = 1 / 60  # 60Hz, in theory
    title = "Oscillate joint angles"

    def viz_update_fn(worlds, _demo_state):
        return

    demo_state = DemoState()
    _run_demo(
        robot,
        1,
        setup_fn,
        loop_fn,
        viz_update_fn,
        time_p_loop=time_p_loop,
        title=title,
        load_terrain=True,
        demo_state=demo_state,
    )
