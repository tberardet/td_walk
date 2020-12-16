#!/usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
import json
import numpy as np

import model

def buildController(dic):
    controller_name = dic.get("type")
    params = dic.get("params")
    if controller_name == "PIDController":
        return PIDController(params)
    elif controller_name == "OpenLoopEffortController":
        return OpenLoopEffortController(params)
    elif controller_name == "FeedForwardController":
        return FeedForwardController(params)
    elif controller_name == "OpenLoopPendulumEffortController":
        return OpenLoopPendulumEffortController(params)
    raise RuntimeError("Unknown controller name: {:}".format(controller_name))

def buildRobotController(dic):
    controllers = []
    for entry in dic.get("controllers"):
        controllers.append(buildController(entry))
    return RobotController(controllers)

class Controller():
    """
    Implement a simple 1D controller
    """
    def __init__(self, params):
        self.cmd_max = params["cmd_max"]

    @abstractmethod
    def step(t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        """
        Parameters
        ----------
        measured_pos : float
        measured_vel : float
        ref_pos : float
        ref_vel : float
        ref_acc : float

        Returns
        -------
        cmd : float
            The command computed by the controller
        """

class PIDController(Controller):

    def __init__(self, params):
        """
        Parameters
        ----------
        params: dictionary
            Classic members: kp, kd, ki
        """
        super().__init__(params)
        self.kp = params["kp"]
        self.kd = params["kd"]
        self.ki = params["ki"]
        self.acc = 0
        self.last_t = None

    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        error = (ref_pos - measured_pos)
        vel_error = (ref_vel - measured_vel)
        if self.last_t is not None and abs(self.ki) > 0:
            dt = t - self.last_t
            self.acc += self.ki * error * dt
            max_acc = self.cmd_max / self.ki # max_acc * ki = cmd_max
            self.acc = np.clip(self.acc, -max_acc, max_acc)
        self.last_t = t
        cmd = self.kp * error + self.ki * self.acc + self.kd * vel_error
        clipped_cmd = np.clip(cmd, -self.cmd_max, self.cmd_max)
        return clipped_cmd

class OpenLoopEffortController(Controller):
    """
    An open-loop controller which uses an effort proportional to acceleration
    """
    def __init__(self, params):
        super().__init__(params)
        self.k_acc = params["k_acc"]

    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        cmd = ref_acc * self.k_acc
        return np.clip(cmd, -self.cmd_max, self.cmd_max)

class OpenLoopPendulumEffortController(Controller):
    """
    An open-loop controller which aims at only compensing the gravity
    """
    def __init__(self, params):
        super().__init__(params)
        self.mass = params["mass"]
        self.dist = params["dist"]

    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        cmd = -np.cos(ref_pos) * self.mass * self.dist * 9.81
        cmd += ref_acc * self.mass * self.dist **2 / 2
        return np.clip(cmd, -self.cmd_max, self.cmd_max)

class FeedForwardController(Controller):
    """
    A controller combining a PIDController and a model
    """
    def __init__(self, params):
        super().__init__(params)
        self.model = buildController(params["model"])
        self.model.cmd_max = self.cmd_max
        self.pid = buildController(params["pid"])
        self.model.cmd_max = self.cmd_max

    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        cmd = self.model.step(t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc)
        cmd += self.pid.step(t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc)
        return np.clip(cmd, -self.cmd_max, self.cmd_max)


class RobotController():
    """
    A controller across multiple dimensions
    """
    def __init__(self, controllers):
        self.controllers = controllers

    def step(self,  t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        N = len(self.controllers)
        cmd = np.zeros(N)
        if ref_vel is None:
            ref_vel = np.zeros(N)
        if ref_acc is None:
            ref_acc = np.zeros(N)
        for i in range(N):
            cmd[i] = self.controllers[i].step(t, measured_pos[i], measured_vel[i],
                                              ref_pos[i], ref_vel[i], ref_acc[i])
        return cmd
