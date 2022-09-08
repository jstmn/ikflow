from typing import List, Callable, Any
from time import time, sleep
from dataclasses import dataclass

from src.ik_solvers import GenerativeIKSolver
import config
from src import robots

from klampt.model import coordinates, trajectory
from klampt import vis
from klampt import WorldModel
import numpy as np
import torch
import torch.optim


""" Class contains functions that show generated solutions to example problems

"""


class _3dDemo:
    def __init__(self, ik_solver: GenerativeIKSolver):
        self.ik_solver = ik_solver
        self.robot_model: robots.KlamptRobotModel = self.ik_solver.robot_model
        self.endeff_link_name: str = self.robot_model._klampt_ee_link.getName()

    def _run_demo(
        self,
        n_worlds: int,
        setup_fn: Callable[[List[WorldModel]], None],
        loop_fn: Callable[[List[WorldModel], Any], None],
        viz_update_fn: Callable[[List[WorldModel], Any], None],
        demo_state: Any = None,
        time_p_loop: float = 2.5,
        title="Anonymous demo",
    ):
        """Internal function for running a demo."""

        worlds = [self.robot_model.world_model.copy() for _ in range(n_worlds)]
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

    def visualize_fk(self, solver="klampt"):
        """Set the robot to a random config. Visualize the poses returned by fk"""

        assert solver in ["klampt", "batch_fk"]

        n_worlds = 1
        time_p_loop = 30
        title = "Visualize poses returned by FK"

        def setup_fn(worlds):
            vis.add(f"robot", worlds[0].robot(0))
            vis.setColor((f"robot", self.endeff_link_name), 0, 1, 0, 0.7)

            # Axis
            vis.add("coordinates", coordinates.manager())
            vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
            vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

        from klampt.math import so3

        def loop_fn(worlds, _demo_state):

            x_random = self.robot_model.sample(1)
            q_random = self.robot_model.x_to_qs(x_random)
            worlds[0].robot(0).setConfig(q_random[0])

            if solver == "klampt":
                fk = self.robot_model.forward_kinematics_klampt(x_random, with_l2_loss_pts=True)
                ee_pose = fk[0, 0:7]
                vis.add("ee", (so3.from_quaternion(ee_pose[3:]), ee_pose[0:3]), length=0.15, width=2)

                for j in range(len(self.robot_model.l2_loss_pts)):
                    root_idx = 7 + 7 * j
                    loss_pt_pose = fk[0, root_idx + 0 : root_idx + 7]
                    vis.add(
                        f"l2_loss_pt_{j}",
                        (so3.from_quaternion(loss_pt_pose[3:]), loss_pt_pose[0:3]),
                        length=0.1,
                        width=2,
                    )

            else:
                # (B x 3*(n+1) )
                x_torch = torch.from_numpy(x_random).float().to(config.device)
                fk = self.robot_model.forward_kinematics_batch(x_torch, with_l2_loss_pts=True)
                ee_pose = fk[0, 0:3]
                vis.add("ee", (so3.identity(), ee_pose[0:3]), length=0.15, width=2)
                for j in range(len(self.robot_model.l2_loss_pts)):
                    root_idx = 3 + 3 * j
                    t = fk[0, root_idx + 0 : root_idx + 3]
                    vis.add(f"l2_loss_pt_{j}", (so3.identity(), t), length=0.1, width=2)

        def viz_update_fn(worlds, _demo_state):
            return

        self._run_demo(n_worlds, setup_fn, loop_fn, viz_update_fn, time_p_loop=time_p_loop, title=title)

    def fixed_endpose_oscilate_latent(self):
        """Fixed end pose, oscillate through the latent space"""

        n_worlds = 2
        time_p_loop = 0.01
        title = "Fixed end pose with oscillation through the latent space"

        def setup_fn(worlds):

            vis.add(f"robot_goal", worlds[0].robot(0))
            vis.setColor(f"robot_goal", 0.5, 1, 1, 0)
            vis.setColor((f"robot_goal", self.endeff_link_name), 0, 1, 0, 0.7)

            vis.add(f"robot_1", worlds[1].robot(0))
            vis.setColor(f"robot_1", 1, 1, 1, 1)
            vis.setColor((f"robot_1", self.endeff_link_name), 1, 1, 1, 0.71)

            # Axis
            vis.add("coordinates", coordinates.manager())
            vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
            vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

        target_pose = np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0])
        rev_input = torch.zeros(1, self.ik_solver.dim_tot).to(config.device)

        @dataclass
        class DemoState:
            counter: int

        def loop_fn(worlds, _demo_state):

            # for i in range(self.ik_solver.dim_tot):
            for i in range(5):
                rev_input[0, i] = 0.25 * np.cos(_demo_state.counter / 25) - 0.1 * np.cos(_demo_state.counter / 250)

            # Get solutions to pose of random sample
            sampled_solutions = self.ik_solver.make_samples(target_pose, 1, latent_noise=rev_input)[0]
            qs = self.robot_model.x_to_qs(sampled_solutions)
            worlds[1].robot(0).setConfig(qs[0])

            # Update _demo_state
            _demo_state.counter += 1

        def viz_update_fn(worlds, _demo_state):
            return

        demo_state = DemoState(counter=0)
        self._run_demo(
            n_worlds, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=time_p_loop, title=title
        )

    # TODO(@jeremysm): Add/flesh out plots. Consider plotting each solutions x, or error
    def oscillating_target_pose(self, nb_sols=5, fixed_latent_noise=True):
        """Oscillating target pose"""

        initial_target_pose = np.array([0, 0.5, 0.25, 1.0, 0.0, 0.0, 0.0])
        time_p_loop = 0.01
        title = "Solutions for oscillating target pose"
        latent = None
        if fixed_latent_noise:
            latent = torch.randn((nb_sols, self.ik_solver.dim_tot)).to(config.device)

        def setup_fn(worlds):
            vis.add("coordinates", coordinates.manager())
            for i in range(len(worlds)):
                vis.add(f"robot_{i}", worlds[i].robot(0))
                vis.setColor(f"robot_{i}", 1, 1, 1, 1)
                vis.setColor((f"robot_{i}", self.endeff_link_name), 1, 1, 1, 0.71)

            # Axis
            vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
            vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))

            # Add plot
            vis.addPlot("target_pose")
            vis.logPlot("target_pose", "target_pose x", 0)
            vis.setPlotDuration("target_pose", 10)

        @dataclass
        class DemoState:
            counter: int
            target_pose: np.ndarray

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
            sampled_solutions = self.ik_solver.make_samples(_demo_state.target_pose, nb_sols, latent_noise=latent)[0]
            qs = self.robot_model.x_to_qs(sampled_solutions)
            for i in range(nb_sols):
                worlds[i].robot(0).setConfig(qs[i])

            # Update _demo_state
            _demo_state.counter += 1

        def viz_update_fn(worlds, _demo_state):
            line = trajectory.Trajectory([1, 0], [_demo_state.target_pose[0:3], [0, 0, 0]])
            vis.add("target_pose.", line)

            vis.logPlot("target_pose", "target_pose x", _demo_state.target_pose[0])

        demo_state = DemoState(counter=0, target_pose=initial_target_pose.copy())
        self._run_demo(
            nb_sols, setup_fn, loop_fn, viz_update_fn, demo_state=demo_state, time_p_loop=time_p_loop, title=title
        )

    def random_target_pose(self, nb_sols=5):
        """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""

        def setup_fn(worlds):
            vis.add(f"robot_goal", worlds[0].robot(0))
            vis.setColor(f"robot_goal", 0.5, 1, 1, 0)
            vis.setColor((f"robot_goal", self.endeff_link_name), 0, 1, 0, 0.7)

            for i in range(1, nb_sols + 1):
                vis.add(f"robot_{i}", worlds[i].robot(0))
                vis.setColor(f"robot_{i}", 1, 1, 1, 1)
                vis.setColor((f"robot_{i}", self.endeff_link_name), 1, 1, 1, 0.71)

        def loop_fn(worlds, _demo_state):
            # Get random sample
            random_sample = self.robot_model.sample(1)
            random_sample_q = self.robot_model.x_to_qs(random_sample)
            worlds[0].robot(0).setConfig(random_sample_q[0])
            target_pose = self.robot_model.forward_kinematics_klampt(random_sample)[0]

            # Get solutions to pose of random sample
            sampled_solutions = self.ik_solver.make_samples(target_pose, nb_sols)[0]
            qs = self.robot_model.x_to_qs(sampled_solutions)
            for i in range(nb_sols):
                worlds[i + 1].robot(0).setConfig(qs[i])

        time_p_loop = 2.5
        title = "Solutions for randomly drawn poses - Green link is the target pose"

        def viz_update_fn(worlds, _demo_state):
            return

        self._run_demo(nb_sols + 1, setup_fn, loop_fn, viz_update_fn, time_p_loop=time_p_loop, title=title)

    def oscillate_joints(self):
        """Set the end effector to a randomly drawn pose. Generate and visualize `nb_sols` solutions for the pose"""

        def setup_fn(worlds):
            vis.add("robot", worlds[0].robot(0))
            vis.setColor("robot", 1, 1, 1, 1)

        def loop_fn(worlds, _demo_state):

            # Get random sample
            random_sample = self.robot_model.sample(1)
            q = self.robot_model.x_to_qs(random_sample)
            worlds[0].robot(0).setConfig(q[0])

        time_p_loop = 2.5
        title = "Oscillate joint angles"

        def viz_update_fn(worlds, _demo_state):
            return

        self._run_demo(1, setup_fn, loop_fn, viz_update_fn, time_p_loop=time_p_loop, title=title)
