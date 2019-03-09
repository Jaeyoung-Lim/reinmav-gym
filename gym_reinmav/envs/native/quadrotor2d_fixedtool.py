#Copyright (C) 2018, by Jaeyoung Lim, jalim@ethz.ch
# 2D quadrotor environment with a fixed tool attach

import gym
from gym import error, spaces, utils, logger
from math import cos, sin, pi, atan2
import numpy as np
from numpy import linalg
from gym.utils import seeding

class Quadrotor2DFixedtool(gym.Env):
	metadata = {'render.modes': ['human']}
	"""
	Description:
		A Quadrotor model which is free to move in 2 dimensional space. The goal is to stay as close as the reference position
	Observation: 
		Type: Box(5)
		Num	Observation                 			Min         Max
		0	Tool Error Position x             -3.0         3.0
		1	Tool Error Position z             -3.0         3.0
		2	Quadrotor Attitude                 -180 deg        180 deg
		3	Tool Error Velocity x            -10.0        10.0
		4	Tool Error Velocity z            -10.0        10.0
	
	Actions:
		Type: Discrete(2)
		Num	Action
		0	Thrust
		1	Body Rate
	Source: 
		Nguyen, Hai-Nguyen, ChangSu Ha, and Dongjun Lee. "Mechanics, control and internal dynamics of quadrotor tool operation." Automatica 61 (2015): 289-301.	
	Reward:
		Reward is the negative of the distance towards the reference
	Starting State:
		All observations are assigned a uniform random value in [-3.0..3.0]
	Episode Termination:
		The vehicle is outside the bounds of state space
		The vehicle velocity is outside the bounds of state space
	"""
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, -9.8])

		self.state = None
		self.ref_pos = np.array([0.0, 0.0])
		self.ref_vel = np.array([0.0, 0.0])
		self.tool_pos = np.array([0.5, 0.5])

		# Conditions to fail the episode
		self.pos_threshold = 3.0
		self.vel_threshold = 10.0

		self.viewer = None
		self.quadtrans = None
		self.reftrans = None
		self.x_range = 1.0
		self.steps_beyond_done = None

		self.action_space = spaces.Box(low=-10.0, high=10.0,
                                       shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                        shape=(5,), dtype=np.float32)
		self.seed()
		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1] # Angular velocity command

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos_quad = np.array([state[0], state[1]]).flatten()
		att = np.array([state[2]]).flatten()
		vel = np.array([state[3], state[4]]).flatten()
		pos = pos_quad + np.array([self.tool_pos[0]*cos(att) - self.tool_pos[1]*sin(att), self.tool_pos[0]*sin(att) + self.tool_pos[1]*cos(att)])

		acc = thrust/self.mass * np.array([cos(att + pi/2), sin(att + pi/2)]) + self.g
		pos = pos + vel * self.dt + 0.5*acc*self.dt*self.dt
		vel = vel + acc * self.dt
		att = att + w * self.dt

		pos_quad = pos - np.array([self.tool_pos[0]*cos(att) - self.tool_pos[1]*sin(att), self.tool_pos[0]*sin(att) + self.tool_pos[1]*cos(att)])

		self.state = (pos_quad[0], pos_quad[1], att, vel[0], vel[1])

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
		Kp = -5.0
		Kv = -4.0
		tau = 0.1
		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1]]).flatten()
		att = np.array([state[2]]).flatten()
		vel = np.array([state[3], state[4]]).flatten()

		error_pos = pos - ref_pos
		error_vel = vel - ref_vel
		# Calculate desired acceleration
		desired_acc = Kp * error_pos + Kv * error_vel + np.array([0.0, 9.8])
		desired_att = atan2(desired_acc[1], desired_acc[0]) - pi/2
		error_att = att - desired_att
		w = (-1/tau) * error_att
		thrust = self.mass * linalg.norm(desired_acc, 2)

		action = np.array([thrust, w])

		return action 

	def reset(self):
		print("reset")
		self.state = np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(5,)))
		return np.array(self.state)

	def render(self, mode='human', close=False):
		screen_width = 600
		screen_height = 400

		world_width = self.x_range*2
		scale = screen_width/world_width
		quadwidth = 80.0
		quadheight = 10.0
		ref_size = 5.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			# Draw drone
			l,r,t,b = -quadwidth/2, quadwidth/2, quadheight/2, -quadheight/2
			quad = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.quadtrans = rendering.Transform()
			quad.add_attr(self.quadtrans)
			self.viewer.add_geom(quad)
			# Draw refereence
			ref = rendering.make_circle(ref_size)
			self.reftrans = rendering.Transform()
			ref.add_attr(self.reftrans)
			ref.set_color(1,0,0)
			self.viewer.add_geom(ref)

		if self.state is None: return None

		state = self.state
		x = np.array([state[0], state[1]]).flatten()
		theta = self.state[2]
		quad_x = x[0]*scale+screen_width/2.0 
		quad_y = x[1]*scale+screen_height/2.0 
		self.quadtrans.set_translation(quad_x, quad_y)
		self.quadtrans.set_rotation(theta)

		y = self.ref_pos
		ref_x = y[0]*scale+screen_width/2.0
		ref_y = y[1]*scale+screen_height/2.0
		self.reftrans.set_translation(ref_x, ref_y)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
	
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None