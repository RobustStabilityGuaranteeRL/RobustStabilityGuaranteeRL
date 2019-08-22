"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from copy import copy

class CartPoleEnv_adv(gym.Env):


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.cost_weitht = np.array([1., 10., 0., 0., 0.])
        self.gravity = 9.8066
        # 1 0.1 0.5 original
        self.masscart = 0.2275
        self.masspole = 0.0923
        self.total_mass = (self.masspole + self.masscart)
        self.friction_coef = 0.1
        self.joint_friction_coef = 0.01
        # self.observe_noise_scale = np.array([0.005, 0.01, 0.02, 0.01])
        self.observe_noise_scale = np.array([0, 0.00, 0.00, 0.00])
        self.length = 0.185  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 9.8 * 2.6 / 1.75
        self.max_duty_ratio = 4500
        self.tau = 0.01  # seconds between state updates
        self.kinematics_integrator = 'friction'
        self.cons_pos = 4
        self.target_pos = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 30 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.20
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50


        ## motor parameter
        V_s = 24  # nominal voltage
        I_s = 7  # stall current
        self.R = R = V_s / I_s  ## Resistance
        I_n = 2.3  ## Nominal current
        T_n = 2.6 * 9.8 / 100  # Nominal torque  N*M
        self.k_t = T_n / I_n
        I_no_load = 0.3  ## No load current
        w_no_load = 1690 * 2 * np.pi / 60  ## rad/s no load speed
        w_nominal = 1200 * 2 * np.pi / 60  ## rad/s nominal speed
        self.k_i = (V_s - I_no_load * R) / w_no_load
        # k_i = (V_s - I_n * R) / w_nominal
        self.r = 1.75e-2  # radial of motor m
        self.reduce_speed_ratio = 5.18

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_v,
            self.theta_threshold_radians * 2,
            self.max_w])

        self.action_space = spaces.Box(low=-self.max_duty_ratio, high=self.max_duty_ratio, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        self.gravity = gravity
        self.length = length
        self.masspole = mass_of_pole
        self.masscart = mass_of_cart
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def get_params(self):

        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):

        self.gravity = 10
        self.masscart = 1
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def observe(self, s):
        noise = np.random.normal(0., 1., 4) * self.observe_noise_scale
        observation = copy(s) + noise
        return observation

    def step(self, action, impulse=0, process_noise=np.zeros([5])):
        a = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # self.gravity = np.random.normal(10, 2)
        # self.masscart = np.random.normal(1, 0.2)
        # self.masspole = np.random.normal(0.1, 0.02)


        state = self.state
        x, x_dot, theta, theta_dot = state

        voltage = action / 7200 * 12
        motor_speed = x_dot/self.r * self.reduce_speed_ratio
        self.I = (voltage - self.k_i * motor_speed)/self.R
        control_force = self.k_t * self.I /self.r
        force = np.random.normal(control_force, 0)# wind
        force = force + process_noise[0] + impulse
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force-self.friction_coef * x_dot + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - self.joint_friction_coef * theta_dot - costheta * temp) / (
                self.length * (1. - self.masspole * costheta * costheta / self.total_mass))

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot + process_noise[2]
        x_dot = x_dot + self.tau * xacc + process_noise[4]
        # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
        theta = theta + self.tau * theta_dot + process_noise[1]
        theta_dot = theta_dot + self.tau * thetaacc + process_noise[3]
        # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w):

        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        observation = self.observe(self.state)
        # observation = self.state
        done = abs(x) > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians \
               or abs(x_dot) > self.max_v \

        done = bool(done)
        if x < -self.x_threshold \
                or x > self.x_threshold:
            a = 1
        # r1 = ((self.x_threshold/10 - abs(x-self.target_pos))) / (self.x_threshold/10)  # -4-----1
        # r2 = ((self.theta_threshold_radians / 4) - abs(theta)) / (self.theta_threshold_radians / 4)  # -3--------1
        # r1 = max(10 * (1 - ((x-self.target_pos)/self.x_threshold) **2), 1)
        # r2 = max(10 * (1 - np.abs((theta)/self.theta_threshold_radians)), 1)
        # cost1=(self.x_threshold - abs(x))/self.x_threshold
        # e1 = (abs(x)) / self.x_threshold
        # e2 = (abs(theta)) / self.theta_threshold_radians
        # cost = COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot)
        # cost = 0.1+10*max(0, (self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians) \
        #     #+ 5*max(0, (self.x_threshold - abs(x-self.target_pos))/self.x_threshold)\
        cost = np.sum(self.cost_weitht * np.array([x**2/self.x_threshold**2,
                                                   (theta/ self.theta_threshold_radians)**2,
                                                   abs(x_dot[0]),
                                                   abs(theta_dot[0]),
                                                   (action[0] / self.max_duty_ratio) ** 2]))

        # if done:
        #     cost = 100.
        l_rewards = 0
        if abs(x)>self.cons_pos:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0
        return observation, cost, done, dict(hit=a,
                                            l_rewards=l_rewards,
                                            cons_pos=self.cons_pos,
                                            cons_theta=self.theta_threshold_radians,
                                            target=self.target_pos,
                                            violation_of_constraint=violation_of_constraint
                                            )

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state [1] = 0.
        self.state[3] = 0.
        # self.state[0] = self.np_random.uniform(low=5, high=6)
        self.state[0] = self.np_random.uniform(low=-self.x_threshold/2, high=self.x_threshold/2)
        # self.state[2] = self.np_random.uniform(low=-self.theta_threshold_radians / 4, high=self.theta_threshold_radians / 4)
        self.steps_beyond_done = None
        # self.friction_coef = np.random.normal(0.1, 0.05)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Render the target position
            self.target = rendering.Line((self.target_pos * scale + screen_width / 2.0, 0),
                                         (self.target_pos * scale + screen_width / 2.0, screen_height))
            self.target.set_color(1, 0, 0)
            self.viewer.add_geom(self.target)


            # # Render the constrain position
            # self.cons = rendering.Line((self.cons_pos * scale + screen_width / 2.0, 0),
            #                              (self.cons_pos * scale + screen_width / 2.0, screen_height))
            # self.cons.set_color(0, 0, 1)
            # self.viewer.add_geom(self.cons)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def COST_1000(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - 4 * abs(x) ** 2
    return cost

def COST_V3(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - abs(x) ** 4
    return cost

def COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 20*np.sign(r2) * ((r2) ** 2)+ 1* np.sign(r1) * (( r1) ** 2)
    return cost


def COST_V2(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 5 * max(r2, 0) + 1* max(r1,0) + 1
    return cost
