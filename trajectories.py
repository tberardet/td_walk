#!/usr/bin/env python3

import numpy as np
import json
import math
import argparse
from abc import ABC, abstractmethod
import traceback

import model

def buildTrajectory(type_name, start, knots, parameters = None):
    if type_name == "ConstantSpline":
        return ConstantSpline(knots, start)
    if type_name == "LinearSpline":
        return LinearSpline(knots, start)
    if type_name == "CubicZeroDerivativeSpline":
        return CubicZeroDerivativeSpline(knots, start)
    if type_name == "CubicWideStencilSpline":
        return CubicWideStencilSpline(knots, start)
    if type_name == "CubicCustomDerivativeSpline":
        return CubicCustomDerivativeSpline(knots, start)
    if type_name == "NaturalCubicSpline":
        return NaturalCubicSpline(knots, start)
    if type_name == "PeriodicCubicSpline":
        return PeriodicCubicSpline(knots, start)
    if type_name == "TrapezoidalVelocity":
        if parameters is None:
            raise RuntimeError("Parameters can't be None for TrapezoidalVelocity")
        return TrapezoidalVelocity(knots, parameters["vel_max"], parameters["acc_max"], start)
    raise RuntimeError("Unknown type: {:}".format(type_name))

def buildTrajectoryFromDictionary(dic):
    return buildTrajectory(dic["type_name"], dic["start"], np.array(dic["knots"]), dic.get("parameters"))

def buildRobotTrajectoryFromDictionary(dic):
    robot = model.getRobotModel(dic["model_name"])
    return RobotTrajectory(robot, np.array(dic["targets"]), dic["trajectory_type"],
                           dic.get("start"), dic.get("parameters"))

def buildWalkEngineFromDictionary(dic):
    robot = model.getRobotModel(dic["model_name"])
    return WalkEngine(robot, dic.get("start"), dic.get("parameters"))


def getDerivationCoeff(k,d):
    mult = 1
    for i in range(d):
        mult *= (k-i)
    return mult

def getAOPDerivative(t, D, k):
    """
    Compute the k-th derivative of the All-One-Polynomial of degree D

    Parameters
    ----------
    t : float
        The time at which the array is requested
    D : int
        The order of the polynome
    k : int
        The order of the derivative
    Returns
    -------
    v : np.array shape(nb_elements,)
        Return the k-th derivative of the array [t**D, t**(D-1), ..., 1]
    """
    A = np.zeros(D+1)
    if (k > D):
        return A
    for i in range(D-k):
        mult = getDerivationCoeff(D-i,k)
        A[i] = mult * t ** (D-k-i)
    A[D-k] = getDerivationCoeff(k,k)
    return A

class Trajectory:
    """
    Describe a one dimension trajectory. Provides access to the value
    for any degree of the derivative

    Parameters
    ----------
    start: float
        The time at which the trajectory starts
    end: float or None
        The time at which the trajectory ends, or None if the trajectory never
        ends
    """
    def __init__(self, start = 0):
        """
        The child class is responsible for setting the attribute end, if the
        trajectory is not periodic
        """
        self.start = start
        self.end = None

    @abstractmethod
    def getVal(self, t, d):
        """
        Computes the value of the derivative of order d at time t.

        Notes:
        - If $t$ is before (resp. after) the start (resp. after) of the
         trajectory, returns:
          - The position at the start (resp. end) of the trajectory if d=0
          - 0 for any other value of $d$

        Parameters
        ----------
        t : float
            The time at which the position is requested
        d : int >= 0
            Order of the derivative. 0 to access position, 1 for speed,
            2 for acc, etc...

        Returns
        -------
        x : float
            The value of derivative of degree d at time t.
        """

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    def getDuration(self):
        return self.end - self.start

class Spline(Trajectory):
    """
    Attributes
    ----------
    knots : np.ndarray shape (N,2+)
        The list of timing for all the N via points:
        - Column 0 represents time points
        - Column 1 represents the position
        - Additional columns might be used to specify other elements
          (e.g derivative)
    coeffs : np.ndarray shape(N-1,K+1)
        A list of n-1 polynomials of degree $K$: The polynomial at slice $i$ is
        defined as follows: $S_i(t) = \sum_{j=0}^{k}coeffs[i,j] * (t-t_i)^(k-j)$
    """

    def __init__(self, knots, start = 0):
        super().__init__(start)
        if np.min(knots[1:,0] - knots[:-1,0]) <= 0:
            raise RuntimeError("Values are not properly ordered")
        self.knots = knots.copy()
        self.end = knots[-1,0] + self.start
        self.updatePolynomials()

    @abstractmethod
    def updatePolynomials(self):
        """
        Updates the polynomials based on the knots and the interpolation method
        """

    def getDegree(self):
        """
        Returns
        -------
        d : int
            The degree of the polynomials used in this spline
        """
        return self.coeffs.shape[1] - 1

    def getPolynomial(self, t):
        """
        Parameters
        ----------
        t : float
           The time at which the polynomial is requested

        Returns
        -------
        adjusted_t : float
            Normalized time for the slice considered
        p : np.ndarray shape(k+1,)
            The coefficients of the polynomial at time t, see coeffs
        """
        t -= self.start
        # For trajectories with a large number of points, dichotomic search
        # would be appreciated
        for i in range(1,self.knots.shape[0]):
            if self.knots[i,0] > t:
                return t - self.knots[i-1,0], self.coeffs[i-1,:]
        return t - self.knots[-1,0], self.coeffs[-1,:]

    def getVal(self, t, d = 0):
        if t >= self.knots[-1,0] + self.start:
            if d == 0:
                return self.knots[-1,1]
            return 0
        if (t < self.knots[0,0] + self.start) and d > 0:
            return 0
        t = max(self.knots[0,0]+self.start,t)
        adjusted_t, p  = self.getPolynomial(t)
        T = getAOPDerivative(adjusted_t, p.shape[0]-1, d)
        return p @ T

class ConstantSpline(Spline):
    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0]-1,1))
        self.coeffs[:,0] = self.knots[:-1,1]

class LinearSpline(Spline):
    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0]-1,2))
        self.coeffs[:,1] = self.knots[:-1,1]
        dt = self.knots[1:,0] - self.knots[:-1,0]
        dval = self.knots[1:,1] - self.knots[:-1,1]
        self.coeffs[:,0] = dval / dt

class CubicZeroDerivativeSpline(Spline):
    """
    Update polynomials ensuring derivative is 0 at every knot.
    """

    def updatePolynomials(self):
        if self.knots.shape[1] != 2:
            raise RuntimeError("CubicSpline with zero derivative uses only time"
                               " and position as knots")
        nb_pol = self.knots.shape[0]-1
        self.coeffs = np.zeros((nb_pol,4))
        for i in range(nb_pol):
            # Building equation system with:
            # 0: s_i(t_i) = x_i
            # 1: s_i(t_{i+1}) = x_{i+1}
            # 2: s_i'(t_i) = 0
            # 3: s_i'(t_i+1) = 0
            A = np.zeros((4,4))
            B = np.zeros(4)
            B[:2] = self.knots[i:i+2,1]
            local_t = self.knots[i:i+2,0] - self.knots[i,0]
            A[0] = getAOPDerivative(local_t[0],3,0)
            A[1] = getAOPDerivative(local_t[1],3,0)
            A[2] = getAOPDerivative(local_t[0],3,1)
            A[3] = getAOPDerivative(local_t[1],3,1)
            self.coeffs[i,:] = np.linalg.solve(A,B)

class CubicWideStencilSpline(Spline):
    """
    Update polynomials based on a larger neighborhood following the method 1
    described in http://www.math.univ-metz.fr/~croisil/M1-0809/2.pdf
    """

    def updatePolynomials(self):
        if self.knots.shape[0] < 4:
            raise RuntimeError("CubicSpline with wide stencil requires at least"
                               " 4 points.")
        if self.knots.shape[1] != 2:
            raise RuntimeError("CubicSpline with wide stencil uses only time"
                               " and position as knots")

        nb_pol = self.knots.shape[0]-1
        self.coeffs = np.zeros((nb_pol,4))
        for i in range(0,self.knots.shape[0]-1):
            A = np.zeros((4,4))
            idx_start = i-1
            idx_end = i+3
            if i == 0:
                idx_start = 0
                idx_end = 4
            elif i == self.knots.shape[0]-2:
                idx_start = i-2
                idx_end = i+2
            B = self.knots[idx_start:idx_end,1]
            local_t = self.knots[idx_start:idx_end,0] - self.knots[i,0]
            for r in range(4):
                A[r,:] = getAOPDerivative(local_t[r],3,0)
            self.coeffs[i,:] = np.linalg.solve(A,B)

class CubicCustomDerivativeSpline(Spline):
    """
    For this splines, user is requested to specify the velocity at every knot.
    Therefore, knots is of shape (N,3)
    """
    def updatePolynomials(self):
        if self.knots.shape[1] != 3:
            raise RuntimeError("CubicSpline with custom derivative uses"
                               "(time,pos,vel) as knots")
        nb_pol = self.knots.shape[0]-1
        self.coeffs = np.zeros((nb_pol,4))
        for i in range(nb_pol):
            # Building equation system with:
            # 0: s_i(t_i) = x_i
            # 1: s_i(t_{i+1}) = x_{i+1}
            # 2: s_i'(t_i) = v_i
            # 3: s_i'(t_i+1) = v_{i+1}
            A = np.zeros((4,4))
            B = np.zeros(4)
            local_t = self.knots[i:i+2,0] - self.knots[i,0]
            for k in range(2):
                A[k] = getAOPDerivative(local_t[k],3,0)
                B[k] = self.knots[i+k,1]
                A[2+k] = getAOPDerivative(local_t[k],3,1)
                B[2+k] = self.knots[i+k,2]
            self.coeffs[i,:] = np.linalg.solve(A,B)

class NaturalCubicSpline(Spline):
    def updatePolynomials(self):
        if self.knots.shape[1] != 2:
            raise RuntimeError("NaturalCubicSpline uses (time,pos) as knots")
        nb_pol = self.knots.shape[0]-1
        nb_eq = 4 * nb_pol
        A = np.zeros((nb_eq,nb_eq))
        B = np.zeros(nb_eq)
        for i in range(nb_pol):
            dt = self.knots[i+1,0] -self.knots[i,0]
            coef_src = 4*i
            coef_next = 4*(i+1)
            # Position constraints
            A[4*i,coef_src:coef_next] = getAOPDerivative(0,3,0)
            B[4*i] = self.knots[i,1]
            A[4*i+1,coef_src:coef_next] = getAOPDerivative(dt,3,0)
            B[4*i+1] = self.knots[i+1,1]
            # Velocity constraints
            if (i < nb_pol -1):
                A[4*i+2,coef_src:coef_next] = getAOPDerivative(dt,3,1)
                A[4*i+2,coef_next:(coef_next+4)] = -getAOPDerivative(0,3,1)
                A[4*i+3,coef_src:coef_next] = getAOPDerivative(dt,3,2)
                A[4*i+3,coef_next:(coef_next+4)] = -getAOPDerivative(0,3,2)
        # Additional constraints: start and end with 0 2nd derivative
        A[-2,:4] = getAOPDerivative(0,3,2)
        last_dt = self.knots[-1,0] - self.knots[-2,0]
        A[-1,-4:] = getAOPDerivative(last_dt,3,2)
        coeffs = np.linalg.solve(A,B)
        self.coeffs = np.zeros((nb_pol,4))
        for i in range(nb_pol):
            self.coeffs[i,:] = coeffs[4*i:4*(i+1)]

class PeriodicCubicSpline(Spline):
    """
    Describe global splines where position, 1st order derivative and second
    derivative are always equal on both sides of a knot. This i
    """
    def updatePolynomials(self):
        if self.knots.shape[1] != 2:
            raise RuntimeError("PeriodicCubicSpline uses (time,pos) as knots")
        if self.knots[-1,1] != self.knots[0,1]:
            raise RuntimeError("PeriodicCubicSpline requires first pos and last"
                               " pos to be equal")
        nb_pol = self.knots.shape[0]-1
        nb_eq = 4 * nb_pol
        A = np.zeros((nb_eq,nb_eq))
        B = np.zeros(nb_eq)
        for i in range(nb_pol):
            dt = self.knots[i+1,0] -self.knots[i,0]
            coef_src = 4*i
            coef_next = 4*(i+1)
            # Position constraints
            A[4*i,coef_src:coef_next] = getAOPDerivative(0,3,0)
            B[4*i] = self.knots[i,1]
            A[4*i+1,coef_src:coef_next] = getAOPDerivative(dt,3,0)
            B[4*i+1] = self.knots[i+1,1]
            # Velocity constraints
            if (i < nb_pol -1):
                A[4*i+2,coef_src:coef_next] = getAOPDerivative(dt,3,1)
                A[4*i+2,coef_next:(coef_next+4)] = -getAOPDerivative(0,3,1)
                A[4*i+3,coef_src:coef_next] = getAOPDerivative(dt,3,2)
                A[4*i+3,coef_next:(coef_next+4)] = -getAOPDerivative(0,3,2)
        # Additional constraints: start and end have same 1st and 2nd degree derivative
        last_dt = self.knots[-1,0] - self.knots[-2,0]
        for k in [1,2]:
            A[-k,:4] = getAOPDerivative(0,3,k)
            A[-k,-4:] = -getAOPDerivative(last_dt,3,k)
        coeffs = np.linalg.solve(A,B)
        self.coeffs = np.zeros((nb_pol,4))
        for i in range(nb_pol):
            self.coeffs[i,:] = coeffs[4*i:4*(i+1)]

    def getVal(self, t, d = 0):
        duration = self.end - self.start
        normalized_t = (t-self.start) % duration
        return super().getVal(self.start + normalized_t, d)

class TrapezoidalVelocitySlice(Trajectory):
    def __init__(self, xSrc, xEnd, vMax, accMax, start):
        super().__init__(start)
        self.xSrc = xSrc
        self.xEnd = xEnd
        self.vMax = vMax
        self.accMax = accMax
        self.D = xEnd - xSrc
        self.accTime = vMax / accMax
        if abs(self.D) < vMax ** 2 / accMax:
            self.accTime = math.sqrt(abs(self.D)/accMax)
        self.distAcc = accMax * self.accTime ** 2 / 2
        steady_vel_time = (abs(self.D) - 2 * self.distAcc) / vMax
        self.end = self.start + 2*self.accTime + steady_vel_time

    def getVal(self, t, d):
        # Trapezoidal law in velocity has always 0 jerk (except on changing points, to be handled)
        if d > 2:
            return 0
        if t < self.start:
            if d == 0:
                return self.xSrc
            return 0
        if t > self.end:
            if d == 0:
                return self.xEnd
            return 0
        t = t - self.start
        duration = self.end - self.start
        signed_vel = np.sign(self.D) * self.vMax
        signed_acc = np.sign(self.D) * self.accMax
        if t <= self.accTime:
            if d == 0:
                return self.xSrc + signed_acc * t**2 / 2
            if d == 1:
                return signed_acc * t
            if d == 2:
                return signed_acc
        if t >= duration - self.accTime:
            to_end = duration - t
            if d == 0:
                return self.xEnd - signed_acc * to_end**2 / 2
            if d == 1:
                return signed_acc * to_end
            if d == 2:
                return -signed_acc
        if d == 0:
            return self.xSrc + np.sign(self.D) * self.distAcc + signed_vel * (t - self.accTime)
        if d == 1:
            return signed_vel
        return 0

class TrapezoidalVelocity(Trajectory):
    def __init__(self, knots, vMax, accMax, start):
        if len(knots.shape) > 1 and knots.shape[1] > 1:
            raise RuntimeError("Invalid dimension for knots")
        self.inner_trajectories = []
        self.start = start
        for i in range(knots.shape[0]-1):
            new_t = TrapezoidalVelocitySlice(knots[i], knots[i+1], vMax, accMax, start)
            self.inner_trajectories.append(new_t)
            start = new_t.getEnd()
        self.end = start

    def getVal(self, t, d):
        for traj in self.inner_trajectories:
            if t < traj.getEnd():
                return traj.getVal(t,d)
        return self.inner_trajectories[-1].getVal(t,d)

class RobotTrajectory:
    """
    Represents a multi-dimensional trajectory for a robot:
    - targets are always provided in planification space
    - planification is always performed in planification space

    Attributes
    ----------
    model : control.RobotModel
        The model used for the robot
    trajectories : list(Trajectory)
        One trajectory per dimension of the planification space
    """

    supported_spaces = ["operational", "joint"]

    def __init__(self, model, targets, trajectory_type,
                 start = None, parameters = None):
        """
        model : model.RobotModel
            The model of the robot concerned by this trajectory
        targets : np.ndarray shape(m,n) or shape(m,n+1)
            The multi-dimensional knots for the trajectories. One row concerns one
            target. Each column concern one of the dimension of the target space.
            For trajectories with specified time points (e.g. splines), the first
            column indicates time point.
        trajectory_type : str
            The name of the class to be used with trajectory
        start : float
            The start of the trajectories [s]
        parameters : dictionary or None
            A dictionary containing extra-parameters for trajectories
        """
        if start is None:
            start = 0
        self.model = model
        # For each dimension, build associated trajectories
        self.trajectories = []
        offset_idx = 0
        time = None
        N = self.model.getOperationalDimensionLimits().shape[0]
        if targets.shape[1] == N+1:
            time = targets[:,0]
            offset_idx = 1
        self.start = start
        self.end = start
        for dim in range(N):
            i = dim + offset_idx
            dim_knots = targets[:,i]
            if time is not None:
                dim_knots = np.zeros((targets.shape[0],2))
                dim_knots[:,0] = time
                dim_knots[:,1] = targets[:,i]
            dim_traj = buildTrajectory(trajectory_type, start, dim_knots, parameters)
            self.end = max(self.end, dim_traj.getEnd())
            self.trajectories.append(dim_traj)

    def getOpVal(self, t, dim, degree):
        return self.trajectories[dim].getVal(t, degree)

    def getVal(self, t, dim, degree, space):
        """
        Parameters
        ----------
        t : float
            The time at which the value is requested
        dim : int
            The dimension index
        degree : int
            The degree of the derivative requested (0 means position)
        space : str
            The space in which the value is requested: 'operational' or 'joint'

        Returns
        -------
        v : float
            The value of derivative of order degree at time t on dimension dim
            of the chosen space
        """
        assert(space in self.supported_spaces)
        if space == "operational":
            return self.getOpVal(t,dim, degree)
        if degree == 0:
            joint_target = self.getJointTarget(t)
            if joint_target is None:
                return None
            return joint_target[dim]
        elif degree == 1:
            joint_vel = self.getJointVelocity(t)
            if joint_vel is None:
                return None
            return joint_vel[dim]
        return None

    def getOperationalTarget(self, t):
        op_target = np.zeros(self.model.getOperationalDimensionLimits().shape[0])
        for dim in range(op_target.shape[0]):
            op_target[dim] = self.getVal(t, dim, 0, "operational")
        return op_target

    def getJointTarget(self, t):
        target = self.getOperationalTarget(t)
        nb_sol, joint_target = self.model.analyticalMGI(target)
        if nb_sol == 0:
            raise RuntimeError("Invalid trajectory")
        return joint_target

    def getOperationalVelocity(self, t):
        op_vel = np.zeros(self.model.getOperationalDimensionLimits().shape[0])
        for dim in range(op_vel.shape[0]):
            op_vel[dim] = self.getOpVal(t, dim, 1)
        return op_vel

    def getJointVelocity(self, t):
        joints = self.getJointTarget(t)
        op_vel = self.getOperationalVelocity(t)
        if joints is None or op_vel is None:
            return None
        return self.model.computeJacobian(joints) @ op_vel

    def getJointAcceleration(self, t):
        return None

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    def getDuration(self):
        return self.end - self.start

class WalkEngine(RobotTrajectory):
    def __init__(self, model, start = 0, parameters = None):
        """
        Parameters
        ----------
        model: LeggedRobot
            The model of the legged robot for which the trajectory is planned
        """
        if start is None:
            start = 0
        self.model = model
        self.start = start
        self.setDicParameters(parameters)

    def getParameters(self):
        return np.array([self.step_length, self.step_lateral] + self.rest_pos.tolist() +
                        [self.step_height, self.flying_ratio, self.period] +
                        self.leg_phase_offsets.tolist())

    def setParameters(self, values):
        self.step_length = values[0]
        self.step_lateral = values[1]
        self.rest_pos = values[2:5]
        self.step_height = values[5]
        self.flying_ratio = values[6]
        self.period = values[7]
        self.leg_phase_offsets = values[8:]
        self.updateInternalValues()

    def getParametersNames(self):
        legOffSet = []
        for i in range(len(self.leg_phase_offsets)): 
            legOffSet.append("leg_phase_offsets" + str(i))
        return ["step_length","step_lateral", "rest_pos_x", "rest_pos_y", "rest_pos_z", "step_height", 
                "flying_ratio", "period"] + legOffSet

    def getParametersLimits(self):
        limitOffset = [0,1]
        ans = [
            [-0.2,0.2],
            [-0.2,0.2],
            [-0.2,0.2],
            [-0.2,0.2],
            [-0.7,-0.4],
            [0,0.2],
            [0,0.5],
            [0.1,10] #period
            ]
        for i in range (len(self.leg_phase_offsets)):
            ans.append(limitOffset)
        return np.array(ans)


    def setDicParameters(self, parameters):
        self.step_lateral = parameters["step_lateral"] if "step_lateral" in parameters else 0 
        self.step_length = parameters["step_length"]
        self.period = parameters["period"]
        self.rest_pos = np.array(parameters["rest_pos"])
        self.flying_ratio = parameters["flying_ratio"]
        self.step_height = parameters["step_height"]
        self.leg_phase_offsets = np.array(parameters["leg_phase_offsets"])
        self.updateInternalValues()

    def updateInternalValues(self):
        self.x_spline = LinearSpline(np.array([
            [0,-self.step_length/2],
            [self.flying_ratio,self.step_length/2],
            [1.0,-self.step_length/2]]))

        self.y_spline = LinearSpline(np.array([
            [0,-self.step_lateral/2],
            [self.flying_ratio,self.step_lateral/2],
            [1.0,-self.step_lateral/2]]))

        self.z_spline = LinearSpline(np.array([
            [0,0],
            [self.flying_ratio/2,self.step_height],
            [self.flying_ratio,0],[1.0,0]]))
        self.end = self.start + 2 * self.period

    def getPhase(self, t):
        # Don't normalize times before start
        normalized_t = (t-self.start) / self.period
        if normalized_t > 0:
            normalized_t = normalized_t % 1.0
        return normalized_t

    def getRestLeg(self):
        return self.rest_pos

    def getOffset(self, leg_idx, dim, phase, degree):
        if phase < 0:
            return 0
        val = 0
        leg_phase = (phase + self.leg_phase_offsets[leg_idx]) % 1.0
        if dim == 2:
            val += self.z_spline.getVal(leg_phase,degree)
        if dim == 0:
            val += self.x_spline.getVal(leg_phase,degree)
        if dim == 1:
            val += self.y_spline.getVal(leg_phase,degree)
        return val

    def getOpVal(self, t, dim, degree):
        leg_idx = dim // self.model.leg_model.getNbJoints()
        leg_dim = dim % self.model.leg_model.getNbJoints()
        phase = self.getPhase(t)
        rest = self.getRestLeg()
        val = 0
        if degree == 0:
            val += rest[leg_dim]
        val += self.getOffset(leg_idx, leg_dim, phase, degree)
        # Accounting for speed factor impact on speed, acc, etc...
        val *= self.period ** (-degree)
        return val

    def getDuration(self):
        return self.base_trajectory.getDuration() / self.speed_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--type", type=str, default = "1D",
                        choices=["1D","robot","walkEngine"],
                        help="The type of trajectory to be loaded")
    parser.add_argument("--degrees",
                        type=lambda s: np.array([int(item) for item in s.split(',')]),
                        default=[0,1,2],
                        help="The degrees of derivative to plot")
    parser.add_argument("trajectories", nargs="+", type=argparse.FileType('r'))
    args = parser.parse_args()
    trajectories = {}
    tmax = 0
    tmin = 10**10
    for t in args.trajectories:
        try:
            if args.type == "robot":
                trajectories[t.name] = buildRobotTrajectoryFromDictionary(json.load(t))
            elif args.type == "walkEngine":
                trajectories[t.name] = buildWalkEngineFromDictionary(json.load(t))
            elif args.type == "1D":
                trajectories[t.name] = buildTrajectoryFromDictionary(json.load(t))
            else:
                raise RuntimeError("Unhandled type {:}".format(args.type))
            tmax = max(tmax, trajectories[t.name].getEnd())
            tmin = min(tmin, trajectories[t.name].getStart())
        except KeyError as e:
            print("Error while building trajectory from file {:}:\n{:}".format(t.name, traceback.format_exc()))
            exit()
    order_names = ["position", "velocity", "acceleration", "jerk"]
    print("source,t,order,variable,value")
    for source_name, trajectory in trajectories.items():
        for t in np.arange(tmin - args.margin, tmax + args.margin, args.dt):
            for degree in args.degrees:
                order_name = order_names[degree]
                if args.type in ["robot", "leggedRobot"]:
                    space_dims = {
                        "joint" : trajectory.model.getJointsNames(),
                        "operational" : trajectory.model.getOperationalDimensionNames()
                    }
                    for space, dim_names in space_dims.items():
                        for dim in range(len(dim_names)):
                            v = trajectory.getVal(t, dim, degree, space)
                            if v is not None:
                                print("{:},{:},{:},{:},{:}".format(source_name,t,order_name,dim_names[dim],v))
                else:
                    v = trajectory.getVal(t,degree)
                    print("{:},{:},{:},{:},{:}".format(source_name,t,order_name,"x",v))

