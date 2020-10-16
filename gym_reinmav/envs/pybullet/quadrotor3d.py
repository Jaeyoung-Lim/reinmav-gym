# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Jaeyoung Lim <jalim@student.ethz.ch>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import gym
from gym import error, spaces, utils, logger
from math import cos, sin, pi, atan2
import numpy as np
from numpy import linalg
from gym.utils import seeding
from pyquaternion import Quaternion

class PybulletQuadrotor3D(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, 0.0, -9.8])

		self.state = None

		self.ref_pos = np.array([0.0, 0.0, 2.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])

		# Conditions to fail the episode
		self.pos_threshold = 3.0
		self.vel_threshold = 10.0

		self.viewer = None
		self.render_quad1 = None
		self.render_quad2 = None
		self.render_rotor1 = None
		self.render_rotor2 = None
		self.render_rotor3 = None
		self.render_rotor4 = None
		self.render_velocity = None
		self.render_ref = None
		self.x_range = 1.0
		self.steps_beyond_done = None

		self.action_space = spaces.Box(low=0.0, high=10.0, dtype=np.float, shape=(4,))
		self.observation_space = spaces.Box(low=-10.0, high=10.0, dtype=np.float, shape=(10,))
		
		self.seed()
		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1:4] # Angular velocity command

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()


		att_quaternion = Quaternion(att)

		acc = thrust/self.mass * att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])) + self.g
		
		pos = pos + vel * self.dt + 0.5*acc*self.dt*self.dt
		vel = vel + acc * self.dt
		
		q_dot = att_quaternion.derivative(w)
		att = att + q_dot.elements * self.dt

		self.state = (pos[0], pos[1], pos[2], att[0], att[1], att[2], att[3], vel[0], vel[1], vel[2])

		done =  linalg.norm(pos, 2) < -self.pos_threshold \
			or  linalg.norm(pos, 2) > self.pos_threshold \
			or linalg.norm(vel, 2) < -self.vel_threshold \
			or linalg.norm(vel, 2) > self.vel_threshold
		done = bool(done)

		if not done:
		    reward = (-linalg.norm(pos, 2))
		elif self.steps_beyond_done is None:
		    # Pole just fell!
		    self.steps_beyond_done = 0
		    reward = 1.0
		else:
		    if self.steps_beyond_done == 0:
			    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
		    self.steps_beyond_done += 1
		    reward = 0.0

		return np.array(self.state), reward, done, {}

	def control(self):
		def acc2quat(desired_acc, yaw): # TODO: Yaw rotation
			zb_des = desired_acc / linalg.norm(desired_acc)
			yc = np.array([0.0, 1.0, 0.0])
			xb_des = np.cross(yc, zb_des)
			xb_des = xb_des / linalg.norm(xb_des)
			yb_des = np.cross(zb_des, xb_des)
			zb_des = zb_des / linalg.norm(zb_des)

			rotmat = np.array([[xb_des[0], yb_des[0], zb_des[0]],
							   [xb_des[1], yb_des[1], zb_des[1]],
							   [xb_des[2], yb_des[2], zb_des[2]]])

			desired_att = Quaternion(matrix=rotmat)

			return desired_att

		Kp = np.array([-5.0, -5.0, -5.0])
		Kv = np.array([-4.0, -4.0, -4.0])
		tau = 0.3

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()

		error_pos = pos - ref_pos
		error_vel = vel - ref_vel

		# %% Calculate desired acceleration
		reference_acc = np.array([0.0, 0.0, 0.0])
		feedback_acc = Kp * error_pos + Kv * error_vel 

		desired_acc = reference_acc + feedback_acc - self.g

		desired_att = acc2quat(desired_acc, 0.0)

		desired_quat = Quaternion(desired_att)
		current_quat = Quaternion(att)

		error_att = current_quat.conjugate * desired_quat
		qe = error_att.elements

		
		w = (2/tau) * np.sign(qe[0])*qe[1:4]

		
		thrust = desired_acc.dot(current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])))
		
		action = np.array([thrust, w[0], w[1], w[2]])

		return action

	def reset(self):
		print("reset")
		self.state = np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(10,)))
		return np.array(self.state)

	def render(self, mode='human', close=False):
		return True

	def close(self):
		if self.viewer:
			self.viewer = None