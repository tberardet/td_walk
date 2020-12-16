import json
import math
import numpy as np
from abc import ABC, abstractmethod
from scipy import optimize

from homogeneous_transform import *

tol = 1e-9

def cosineLaw(x,y, L1, L2):
    """
    Parameters
    ----------
    x : double
    y : double
    L1: double
    L2: double

    Returns
    -------
    solutions : list((double,double))
        The list of couples (alpha, beta) that allows to reach the provided
        target
    """
    solutions = []
    dist = math.sqrt(x**2+y**2)
    if (dist < abs(L1 - L2)) or dist > (L1 + L2):
        return solutions
    phi = math.atan2(y, x)
    alpha = math.acos((L1**2+dist**2 - L2**2) / (2*L1* dist))
    beta = math.acos((L1**2+ L2**2 - dist**2) / (2*L1* L2))
    solutions.append(np.array([phi+alpha, beta - np.pi]))
    if abs(alpha) > tol:
        solutions.append(np.array([ phi-alpha, np.pi - beta]))
    return solutions


class RobotModel:
    def getNbJoints(self):
        """
        Returns
        -------
        length : int
            The number of joints for the robot
        """
        return len(self.getJointsNames())

    @abstractmethod
    def getJointsNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names given to robot joints
        """

    @abstractmethod
    def getJointsLimits(self):
        """
        Returns
        -------
        np.array
            The values limits for the robot joints, each row is a different
            joint, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getOperationalDimensionNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names of the operational dimensions
        """

    @abstractmethod
    def getOperationalDimensionLimits(self):
        """
        Returns
        -------
        limits : np.array(x,2)
            The values limits for the operational dimensions, each row is a
            different dimension, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getBaseFromToolTransform(self, joints):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The transformation matrix from base to tool
        """

    @abstractmethod
    def computeMGD(self, joint):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The coordinate of the effectors in the operational space
        """

    @abstractmethod
    def computeJacobian(self, joints):
        """
        Parameters
        ----------
        joints : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The jacobian of the robot for given joints values
        """

    @abstractmethod
    def analyticalMGI(self, target):
        """
        Parameters
        ----------
        joints : np.arraynd shape(n,)
            The current values of the joints of the robot in joint space
        target : np.arraynd shape(m,)
            The target in operational space

        Returns
        -------
        nb_solutions : int
            The number of solutions for the given target, -1 if there is an
            infinity of solutions
        joint_pos : np.ndarray shape(X,) or None
            One of the joint configuration which allows to reach the provided
            target. If no solution is available, returns None.
        """

    def computeMGI(self, joints, target, method, seed = None):
        """
        Parameters
        ----------
        joints : np.ndarray shape(n,)
            The current position of joints in angular space
        target : np.ndarray shape(m,)
            The target in operational space
        method : str
            The method used to compute MGI, available choices:
            - analyticalMGI
            - jacobianInverse
            - jacobianTransposed
        seed : None or int
            The seed used for inner random components if needed
        """
        if method == "analyticalMGI":
            nb_sols, sol = self.analyticalMGI(target)
            return sol
        elif method == "jacobianInverse":
            return self.solveJacInverse(joints, target, seed)
        elif method == "jacobianTransposed":
            return self.solveJacTransposed(joints, target)
        raise RuntimeError("Unknown method: " + method)

    def solveJacInverse(self, joints, target, seed = None):
        """
        Parameters
        ----------
        joints: np.ndarray shape(n,)
            The initial position for the search in angular space
        target: np.ndarray shape(n,)
            The wished target for the tool in operational space
        seed: None or int
            Since the method comport some random part, the seed can be specified
            to obtain reproductible results.
        """
        max_iterations = 500
        max_step_size = 0.1
        tol = 10**-6
        for i in range(max_iterations):
            pos = self.computeMGD(joints)
            error = target - pos
            if np.linalg.norm(error) < tol:
                break
            try:
                J_inv = np.linalg.inv(self.computeJacobian(joints))
                step = J_inv @ error
                step_size = np.linalg.norm(step)
                if step_size > max_step_size:
                    step = step / step_size * max_step_size
                joints = joints + step
            except np.linalg.LinAlgError:
                noise_level = 1e-1
                joints = joints + np.random.default_rng(seed).uniform(
                    -noise_level, noise_level,
                    joints.shape[0])
        return joints

    def solveJacTransposed(self, joints, target):
        limits = self.getJointsLimits()
        cost_func = lambda x : np.linalg.norm(self.computeMGD(x) - target, 2)
        jac_func = lambda x : - 2 * (self.computeJacobian(x).transpose() @
                                     (target - self.computeMGD(x)))
        res = optimize.minimize(cost_func, joints,
                                jac= jac_func,
                                bounds = optimize.Bounds(limits[:,0], limits[:,1]))
        return res.x

class LegYPP(RobotModel):
    """
    Model a leg with 3 degrees of freedom
    """
    def __init__(self):
        self.L0 = -0.035
        self.L1 = -0.1
        self.L2 = -0.3
        self.L3 = -0.3
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([self.L1,0,0])
        self.T_2_3 = translation([0,0,self.L2,0])
        self.T_3_E = translation([0,0,self.L3,0])

    def getJointsNames(self):
        return ["hip_yaw", "hip_pitch", "knee"]

    def getJointsLimits(self):
        return np.array([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getOperationalDimensionLimits(self):
        # Warning, all signs are negative
        max_xy = -(self.L1 + self.L2 + self.L3)
        min_z = self.L0 + self.L2 + self.L3
        max_z = self.L0 - self.L2 - self.L3
        return np.array([[-max_xy,max_xy],[-max_xy,max_xy],[min_z,max_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ rot_y(joints[1])
        T_2_3 = self.T_2_3 @ rot_y(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        # When X and Y of target are 'almost' zero, there is an infinity of solutions
        singularity = np.linalg.norm(target[:2]) < tol
        # First: use q0 to align target along y-axis:
        # - There's 2 potential solutions:
        theta = 0
        if not singularity:
            theta = math.atan2(target[1], target[0]) - np.pi
        solutions = []
        for q0 in [theta, theta + np.pi]:
            q0 = q0 % (2*np.pi) - np.pi
            target_in_0 = np.zeros(4, dtype=np.double)
            target_in_0[:3] = target
            target_in_0[3] = 1
            # Put target in the proper referential:
            # only 2 rotations and 2 translations remaining
            target_in_2a = (invert_transform(self.T_1_2) @ rot_z(-q0) @
                            invert_transform(self.T_0_1)  @ target_in_0)
            for q12 in cosineLaw(-target_in_2a[2], -target_in_2a[0], -self.L2, -self.L3):
                solutions.append(np.array([q0,q12[0],q12[1]]))
        if len(solutions) == 0:
            return 0, None
        if singularity:
            return -1, solutions[0]
        return len(solutions), solutions[0]

    def computeJacobian(self, joints):
        J = np.zeros((3,3), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:,0] = (self.T_0_1 @ d_rot_z(joints[0]) @ self.T_1_2 @
                  rot_y(joints[1]) @ self.T_2_3 @ rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,1] = (self.T_0_1 @ rot_z(joints[0]) @ self.T_1_2 @
                  d_rot_y(joints[1]) @ self.T_2_3 @ rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,2] = (self.T_0_1 @ rot_z(joints[0]) @ self.T_1_2 @
                  rot_y(joints[1]) @ self.T_2_3 @ d_rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        return J

class LegRPP(RobotModel):
    """
    Model a leg with 3 degrees of freedom
    """
    def __init__(self):
        self.L0 = 0
        self.L1 = -0.1
        self.L2 = -0.3
        self.L3 = -0.3
        self.T_0_1 = translation([0,0,0])
        self.T_1_2 = translation([0,0,self.L1])
        self.T_2_3 = translation([0,0,self.L2,0])
        self.T_3_E = translation([0,0,self.L3,0])

    def getJointsNames(self):
        return ["hip_roll", "hip_pitch", "knee"]

    def getJointsLimits(self):
        return np.array([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getOperationalDimensionLimits(self):
        # Warning, all signs are negative
        max_xy = -(self.L1 + self.L2 + self.L3)
        min_z = self.L0 + self.L2 + self.L3
        max_z = self.L0 - self.L2 - self.L3
        return np.array([[-max_xy,max_xy],[-max_xy,max_xy],[min_z,max_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_x(joints[0])
        T_1_2 = self.T_1_2 @ rot_y(joints[1])
        T_2_3 = self.T_2_3 @ rot_y(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        # When X and Z of target are 'almost' zero, there is an infinity of solutions
        singularity = np.linalg.norm(target[(0,2),]) < tol
        # First: use q0 to align target along y-axis:
        # - There's 2 potential solutions:
        theta = 0
        if not singularity:
            theta = math.atan2(-target[1], target[2])
        solutions = []
        for q0 in [theta, theta + np.pi]:
            q0 = q0 % (2*np.pi) - np.pi
            target_in_0 = np.zeros(4, dtype=np.double)
            target_in_0[:3] = target
            target_in_0[3] = 1
            # Put target in the proper referential:
            # only 2 rotations and 2 translations remaining
            target_in_2a = (invert_transform(self.T_1_2) @ rot_x(-q0) @
                            invert_transform(self.T_0_1)  @ target_in_0)
            for q12 in cosineLaw(-target_in_2a[2], -target_in_2a[0], -self.L2, -self.L3):
                solutions.append(np.array([q0,q12[0],q12[1]]))
        if len(solutions) == 0:
            return 0, None
        if singularity:
            return -1, solutions[0]
        return len(solutions), solutions[0]

    def computeJacobian(self, joints):
        J = np.zeros((3,3), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:,0] = (self.T_0_1 @ d_rot_x(joints[0]) @ self.T_1_2 @
                  rot_y(joints[1]) @ self.T_2_3 @ rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,1] = (self.T_0_1 @ rot_x(joints[0]) @ self.T_1_2 @
                  d_rot_y(joints[1]) @ self.T_2_3 @ rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,2] = (self.T_0_1 @ rot_x(joints[0]) @ self.T_1_2 @
                  rot_y(joints[1]) @ self.T_2_3 @ d_rot_y(joints[2]) @
                  self.T_3_E)[:3,3]
        return J

class LeggedRobot(RobotModel):
    """
    nb_legs : int
    leg_names : list(str)
    leg_model : RobotModel
        The robot model used for each leg
    """
    def __init__(self, leg_names, leg_model):
        self.nb_legs = len(leg_names)
        self.leg_names = leg_names
        self.leg_model = leg_model
        leg_joint_names = self.leg_model.getJointsNames()
        leg_op_names = self.leg_model.getOperationalDimensionNames()
        self.joint_names = []
        self.op_names = []
        for leg_name in leg_names:
            self.joint_names += [leg_name + "_" + name for name in leg_joint_names]
            self.op_names += [leg_name + "_" + name for name in leg_op_names]
        self.joint_limits = np.tile(leg_model.getJointsLimits(), (self.nb_legs,1))
        self.op_limits = np.tile(leg_model.getOperationalDimensionLimits(), (self.nb_legs,1))

    def nbLegs(self):
        return self.nb_legs

    def getNbJoints(self):
        return self.nb_legs * self.leg_model.getNbJoints()

    def getJointsNames(self):
        return self.joint_names

    def getJointsLimits(self):
        return self.joint_limits

    def getOperationalDimensionNames(self):
        return self.op_names

    def getOperationalDimensionLimits(self):
        return self.op_limits

    def getBaseFromToolTransform(self, joints):
        return None

    def computeMGD(self, joints):
        result = np.zeros(self.nb_legs * self.leg_model.getOperationalDimensionLimits().shape[0])
        for i in range(self.nb_legs):
            start = i * self.leg_model.getNbJoints()
            end = start + self.leg_model.getNbJoints()
            result[start:end] = self.leg_model.computeMGD(joints[start:end])
        return result

    def computeJacobian(self, joints):
        N = self.getNbJoints()
        J = np.zeros((N,N))
        for i in range(self.nb_legs):
            start = i * self.leg_model.getNbJoints()
            end = start + self.leg_model.getNbJoints()
            J[start:end,start:end] = self.leg_model.computeJacobian(joints[start:end])
        return J

    def analyticalMGI(self, target):
        result = np.zeros(self.getNbJoints())
        op_dims = self.leg_model.getOperationalDimensionLimits().shape[0]
        glob_sol = 1
        for i in range(self.nb_legs):
            start = i * op_dims
            end = start + op_dims
            nb_sol, leg_mgi = self.leg_model.analyticalMGI(target[start:end])
            # If one leg has no solution, there is no solution at all
            if nb_sol == 0:
                result = None
                glob_sol = 0
                break
            # If one leg has an infinity of solution, there is an infinity of solution
            if nb_sol == -1:
                glob_sol = -1
            if glob_sol != -1:
                glob_sol *= nb_sol
            result[start:end] = leg_mgi
        return glob_sol, result

class QuadrupedYPP(LeggedRobot):

    def __init__(self):
        super().__init__(["left_front","right_front","left_back","right_back"],
                         LegYPP())

class QuadrupedRPP(LeggedRobot):
    def __init__(self):
        super().__init__(["left_front","right_front","left_back","right_back"],
                         LegRPP())

class HexapodRPP(LeggedRobot):
    def __init__(self):
        super().__init__(["left_front","right_front",
                          "left_middle","right_middle",
                          "left_back","right_back"],
                         LegRPP())



def getRobotModel(robot_name):
    robot = None
    if robot_name == "leg_ypp":
        robot = LegYPP()
    if robot_name == "leg_rpp":
        robot = LegRPP()
    elif robot_name == "quadruped_ypp":
        robot = QuadrupedYPP()
    elif robot_name == "quadruped_rpp":
        robot = QuadrupedRPP()
    elif robot_name == "hexapod_rpp":
        robot = HexapodRPP()
    else:
        raise RuntimeError("Unknown robot name: '" + robot_name + "'")
    return robot
