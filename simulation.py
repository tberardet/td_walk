#!/usr/bin/env python3

import argparse
import json
import math
import numpy as np
import pybullet as p
import pybullet_data
import sys
from time import sleep, time
import control as ctrl
import model
import trajectories as traj
import homogeneous_transform as ht


class UserDebugParameter:
    def __init__(self, name, initial_value, limits):
        self.name = name
        self.value = initial_value
        self.id = p.addUserDebugParameter(
            name, limits[0], limits[1], initial_value)

    def read(self, p):
        self.value = p.readUserDebugParameter(self.id)


class Simulation:
    """
    A class interfacing the pybullet simulation and the robot model

    Attributes
    ----------
    mode : str
        The mode used for the simulation, it can be chosen among:
        - mgd :: target is provided in joint space
        - analyticalMGI :: target provided in operational space, joint target is
                           retrieved through analyticalMGI
        - jacobianInverse :: target provided in operational space, joint target is
                             retrieved through jacobianInverse method
        - jacobianTransposed :: target provided in operational space, joint target is
                                retrieved through jacobianTransposed
    robot_name : str
        The name of the robot active in the simulation
    robot_model : model.RobotModel
        The physical model of the robot
    robot : int
        The pybullet identifier for the robot object
    dt : float
        The time for each simulation step
    t : float
        The simulation time elapsed since beginning of the simulation
    log : file stream or None
        The file to which log data are sent, None if log has never been opened
    last_tick : float or None
        Time in seconds since simulation was ticked for last time, None if it
        has never been ticked
    last_model_update_duration : float or None
        Duration of previous step of model update or None if it has not been measured
        yet.
    operational_pos : np.ndarray shape(N,)
        Position of the tool in the operational space, number of dimensions
        depends on the robot_model
    user_parameters : list of UserDebugParameter
        The parameters that can be used for the simulation
    joints : np.ndarray shape(X,)
        The last values of the robot joints
    joints_target : np.ndarray shape(X,)
        The current target values for the robot joints
    operational_target : np.ndarray shape(X,)
        The target in operational space, None if mode is mgd
    trajectory : traj.WalkEngine or None
        The trajectory that should be followed
    joints_measured_vel : np.ndarray shape(N,)
    joints_target_vel : np.ndarray shape(N,)
    joints_target_acc : np.ndarray shape(N,) or None
        Currently only implemented if trajectory planning is done in operational space
    operational_measured_vel : np.ndarray shape(N,)
    operational_target_vel : np.ndarray shape(N,)
    control_mode : str
        How is motor command specified: ["position", "velocity", "effort"]
    controller : None or ctrl.Controller
        If None, command is obtained using directly the reference
    max_efforts : np.ndarray shape(N,)
        The maximal effort allowed for each joint, None if no controller is used
    cmd_pos : np.ndarray shape(N,) or None
        The command for position for each joint, None when control mode is not
        'position'
    cmd_vel : np.ndarray shape(N,) or None
        The command for velocity for each joint, None when control mode is not
        'velocity'
    cmd_efforts : np.ndarray shape(N,) or None
        The torques/forces desired for each joint, None when control mode is not
        'effort'
    cmd_buffer : np.ndarray or None
        Content of the cmd buffer one col = one entry, None if there's no
        controller
    cmd_buffer_len : int
        Length of the cmd buffer in number of elements
    cmd_buffer_index : int
        Current index for the cmd buffer
    """

    def __init__(self, robot_name, log_path, dt, mode, target, trajectory_path,
                 control_mode, controller_path, cmd_buffer_len = 1):
        self.log = None
        self.robot_name = robot_name
        self.dt = dt
        self.mode = mode
        self.joints_target = None
        self.operational_target = None
        self.control_mode = control_mode
        self.robot_model = model.getRobotModel(robot_name)
        self.cmd_buffer_len = cmd_buffer_len
        self.cmd_buffer = np.zeros((self.robot_model.getNbJoints(),cmd_buffer_len))
        self.cmd_buffer_idx = 0
        if trajectory_path is not None:
            with open(trajectory_path) as f:
                self.trajectory = traj.buildWalkEngineFromDictionary(json.load(f))
        else:
            self.trajectory = None
        if controller_path is not None:
            with open(controller_path) as f:
                self.controller = ctrl.buildRobotController(json.load(f))
        else:
            self.controller = None
        self.launchSimulation()
        self.addUserDebugParameters(target)
        self.initMemory()
        self.logStart(log_path)

    def __del__(self):
        if self.log is not None:
            self.log.close()

    def launchSimulation(self):
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        # Loading ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.loadURDF("plane.urdf")
        # Loading robot
        urdf_path = "resources/{:}_robot.urdf".format(self.robot_name)
        print(urdf_path)
        self.robot = p.loadURDF(urdf_path, useFixedBase=False)
        urdf_nb_joints = p.getNumJoints(self.robot)
        robot_model_nb_joints = self.robot_model.getNbJoints()
        if (robot_model_nb_joints != urdf_nb_joints):
            raise RuntimeError("Mismatch in number of joints : urdf has {:} and robot_model has {:}".format(
                urdf_nb_joints, robot_model_nb_joints))

        self.max_efforts = np.zeros(urdf_nb_joints)
        for i in range(self.robot_model.getNbJoints()):
            joint_info = p.getJointInfo(self.robot, i)
            self.max_efforts[i] = joint_info[10]
            # If torque control is used, it is mandatory to disable default
            # pybullet controllers
            if self.control_mode == "effort":
                ctrl_mode = p.TORQUE_CONTROL
                # Letting max force of velocity control to joint_friction acts
                # *roughly* as friction according to doc. However, note that if
                # friction > max_effort, it does not prevent the joint from
                # moving (while it should)
                joint_friction = joint_info[7]
                p.setJointMotorControl2(self.robot, i, p.VELOCITY_CONTROL,
                                        force=joint_friction)

        # Time management
        p.setRealTimeSimulation(0)
        self.t = 0
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

    def initMemory(self):
        self.last_tick = None
        self.last_model_update_duration = None
        self.joints_measured_vel = None
        self.joints_target_vel = None
        self.joints_target_acc = None
        self.operational_pos = None
        self.operational_measured_vel = None
        self.operational_target_vel = None
        self.cmd_pos = None
        self.cmd_vel = None
        self.cmd_efforts = None

    def addUserDebugParametersList(self, names, limits, initial_values = None):
        """
        Add to the list of current user parameters the parameters specified

        Parameters
        ----------
        names: list(str)
            The names of the debug parameters to add
        limits: np.ndarray shape(X,2)
            The limits for the debug parameters, each row concern another
            parameter, first column is min, second column is max
        initial_values: np.ndarray shape(X,) or None
            The initial values for the debug parameters. If None is used, value
            is chosen to be the middle of the limits
        """
        if len(names) != limits.shape[0]:
            print(names)
            print(limits)
            raise RuntimeError("Incompatible sizes for names and limits: " +
                               str(len(names)) + " and " +
                               str(limits.shape[0]))
        if initial_values is not None:
            if initial_values.shape[0] != limits.shape[0]:
                raise RuntimeError("Incompatible sizes for initial_values and limits: " +
                                   str(initial_values.shape[0]) + " and " +
                                   str(limits.shape[0]))
        else:
            initial_values = limits.mean(axis=1)
        for i in range(len(names)):
            self.user_parameters.append(UserDebugParameter(
                names[i], initial_values[i], limits[i,:]))

    def addUserDebugParameters(self, target):
        self.user_parameters = []
        if self.trajectory is not None:
            self.addUserDebugParametersList(
                self.trajectory.getParametersNames(),
                self.trajectory.getParametersLimits(),
                self.trajectory.getParameters()
            )
        elif self.mode == "mgd":
            self.addUserDebugParametersList(
                self.robot_model.getJointsNames(),
                self.robot_model.getJointsLimits(),
                target)
        else:
            print(self.robot_model.getOperationalDimensionLimits())
            self.addUserDebugParametersList(
                self.robot_model.getOperationalDimensionNames(),
                self.robot_model.getOperationalDimensionLimits(),
                target)

    def updateStatus(self):
        """
        Updates joints position and user parameters from simulations
        """
        self.joints = np.zeros(self.robot_model.getNbJoints())
        self.joints_measured_vel = np.zeros(self.robot_model.getNbJoints())
        for i in range(self.robot_model.getNbJoints()):
            js = p.getJointState(self.robot, i)
            self.joints[i] = js[0]
            self.joints_measured_vel[i] = js[1]
        for param in self.user_parameters:
            param.read(p)

    def updateModel(self):
        """
        Updates operational pos, measured vel based on
        current joints position and velocities
        """
        self.operational_pos = self.robot_model.computeMGD(
            self.joints)
        self.operational_measured_vel = self.robot_model.computeJacobian(self.joints) @ self.joints_measured_vel

    def getDebugAsArray(self):
        l = []
        for p in self.user_parameters:
            l.append(p.value)
        return np.array(l)

    def updateTargets(self):
        """
        Update joints_target and operational_target
        """
        if self.trajectory is not None:
            self.trajectory.setParameters(self.getDebugAsArray())
            self.operational_target = self.trajectory.getOperationalTarget(self.t)
            self.joints_target = self.trajectory.getJointTarget(self.t)
            self.joints_target_vel = self.trajectory.getJointVelocity(self.t)
            self.operational_target_vel = self.trajectory.getOperationalVelocity(self.t)
            self.joints_target_acc = self.trajectory.getJointAcceleration(self.t)
        elif self.mode == "mgd":
            self.joints_target = self.getDebugAsArray()
            self.operational_target = None
        else:
            self.operational_target = self.getDebugAsArray()
            self.joints_target = self.robot_model.computeMGI(
                self.joints,
                self.operational_target,
                self.mode)
        # Independently of the mode, target efforts are based on joint space
        if self.controller is not None:
            cmd = self.controller.step(self.t, self.joints,
                                       self.joints_measured_vel,
                                       self.joints_target,
                                       self.joints_target_vel,
                                       self.joints_target_acc)
            self.cmd_buffer[:,self.cmd_buffer_idx] = cmd
            self.cmd_buffer_idx += 1
            if self.cmd_buffer_idx == self.cmd_buffer_len:
                self.cmd_buffer_idx = 0
            cmd = self.cmd_buffer[:,self.cmd_buffer_idx]
            if self.control_mode == "position":
                self.cmd_pos = cmd
            elif self.control_mode == "velocity":
                self.cmd_vel = cmd
            elif self.control_mode == "effort":
                self.cmd_efforts = cmd
            else:
                raise RuntimeError("Unknown control mode '{:}'".format(self.control_mode))
        else:
            if self.control_mode == "position":
                self.cmd_pos = self.joints_target
            elif self.control_mode == "velocity":
                self.cmd_vel = self.joints_target_vel
            elif self.control_mode == "effort":
                raise RuntimeError("Effort mode not supported without controller")
            else:
                raise RuntimeError("Unknown control mode '{:}'".format(self.control_mode))

    def tick(self):
        if self.joints_target is not None:
            dof_indices = list(range(self.robot_model.getNbJoints()))
            if self.control_mode == "position":
                p.setJointMotorControlArray(self.robot, dof_indices,
                                            p.POSITION_CONTROL,
                                            self.cmd_pos,
                                            forces = self.max_efforts)
            elif self.control_mode == "velocity":
                p.setJointMotorControlArray(self.robot, dof_indices,
                                            p.VELOCITY_CONTROL,
                                            targetVelocities = self.cmd_vel,
                                            forces = self.max_efforts)
            elif self.control_mode == "effort":
                # Since pybullet does not use the effort limits in torque
                # control, the forces are clipped right before calling pybullet
                clipped_efforts = np.clip(self.cmd_efforts,
                                          -self.max_efforts, self.max_efforts)
                p.setJointMotorControlArray(self.robot, dof_indices,
                                            p.TORQUE_CONTROL,
                                            forces = clipped_efforts)
            else:
                raise RuntimeError("Unknown control mode '{:}'".format(self.control_mode))

        # Make sure that time spent is not too high
        now = time()
        if not self.last_tick is None:
            tick_time = now - self.last_tick
            sleep_time = self.dt - tick_time
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                print("Time budget exceeded: {:}".format(
                    tick_time), file=sys.stderr)
        self.last_tick = time()
        self.t += self.dt
        p.stepSimulation()

        model_update_start = time()
        self.updateStatus()
        self.updateModel()
        self.updateTargets()
        model_update_end = time()
        self.last_model_update_duration = model_update_end - model_update_start

        self.logStep()

    def logStart(self, path):
        if path is None:
            self.log = None
            return
        self.log = open(path, "w")
        self.log.write("{:},{:},{:},{:},{:}\n".format(
            "source", "t", "order", "variable", "value"))

    def logStep(self):
        # Skip logging step if not activated
        if self.log is None:
            return
        var_names = {
            "operational" : self.robot_model.getOperationalDimensionNames(),
            "joint" : self.robot_model.getJointsNames()
        }
        log_values = {
            "position" : {
                "joint" : {
                    "measure" : self.joints,
                    "target" : self.joints_target,
                    "cmd" : self.cmd_pos
                },
                "operational" : {
                    "measure" : self.operational_pos,
                    "target" : self.operational_target
                }
            },
            "velocity" : {
                "joint" : {
                    "measure" : self.joints_measured_vel,
                    "target" : self.joints_target_vel,
                    "cmd" : self.cmd_vel
                },
                "operational" : {
                    "measure" : self.operational_measured_vel,
                    "target" : self.operational_target_vel
                }
            },
            "acc" : {
                "joint" : {
                    "target" : self.joints_target_acc
                }
            },
            "effort" : {
                "joint" : {
                    "cmd" : self.cmd_efforts
                }
            }
        }
        for order, order_values in log_values.items():
            for space , space_values in order_values.items():
                for source, values in space_values.items():
                    # Skip missing values
                    if values is None:
                        continue
                    for dim in range(values.shape[0]):
                        var_name = var_names[space][dim]
                        self.log.write("{:},{:},{:},{:},{:}\n".format(
                            source, self.t, order, var_name, values[dim]))
        if self.last_model_update_duration is not None:
            self.log.write("measure,{:},position,dt,{:}\n".format(self.t,self.last_model_update_duration))


if __name__ == "__main__":
    # Reading arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--mode", type=str, default="mgd",
                        choices=["mgd","analyticalMGI","jacobianInverse","jacobianTransposed"],
                        help="The target specification mode for the simulator")
    parser.add_argument("--robot", type=str, default="rt",
                        help="The name of the robot to be used: rt, rrr, leg")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Duration of a simulation step [s]")
    parser.add_argument("--duration", type=float, default=-1,
                        help="Duration of the simulation [s]")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to the output log file")
    parser.add_argument("--target",
                        type=lambda s: np.array([float(item) for item in s.split(',')]),
                        default=None,
                        help="The initial target to use ")
    parser.add_argument("--control-mode", type=str, default="position",
                        choices=["position","velocity","effort"],
                        help="The mode used for controlling the joints")
    parser.add_argument("--trajectory", type=str, default=None,
                        help="The path to a file describing a trajectory")
    parser.add_argument("--controller", type=str, default=None,
                        help="The path to a file describing a controller")
    parser.add_argument("--delay", type=int, default=0,
                        choices = range(10),
                        help="Create artificially a delay of nb steps on"
                        " commands received by controller to have something"
                        " more realistic.")
    args = parser.parse_args()

    # Launching simulation
    simulation = Simulation(args.robot, args.log, args.dt, args.mode,
                            args.target, args.trajectory,
                            args.control_mode,
                            args.controller,
                            args.delay+1)
    simulation.updateStatus()
    while args.duration < 0 or simulation.t < args.duration:
        simulation.tick()
        simulation.updateStatus()
