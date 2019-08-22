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
import tensorflow as tf
import os

class CartPoleEnv_adv(gym.Env):


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size = 256,
                 steps=int(1e5),
                 eval=False,
                 validation_data_capacity=0.05,
                 validation_frequency=0.01
                 ):
        ## learning params
        with tf.variable_scope('real_plant'):
            self.sess = tf.Session()
            self.dataset = {'s':[],
                            'a':[],
                            's_':[]}
            self.validation_dataset = copy(self.dataset)
            self.s = tf.placeholder(tf.float32, [None, 4], name='s')
            self.a = tf.placeholder(tf.float32, [None, 1], name='a')
            self.s_ = tf.placeholder(tf.float32, [None, 4], name='s_dot')
            self.lr_holder = tf.placeholder(tf.float32, None, name='lr')
            self.motor_force = self._build_motor_net(self.s, self.a)
            self.x_friction, self.theta_friction = self._build_friction_net(self.s, self.a)
            self.residual = self._build_residual_net(self.s, self.a, name='residual')
            self.lr = learning_rate
            self.steps = steps
            self.batch_size = batch_size
            self.eval = eval
            self.validation_data_capacity = validation_data_capacity
            self.validation_frequency = validation_frequency

            self.cost_weight = np.array([4., 10., 0., 0., 0.])
            self.observe_noise_scale = np.array([0, 0.00, 0.00, 0.00])
            self.gravity = tf.constant([9.8066],name='gravity')
            # 1 0.1 0.5 original
            self.masscart = tf.constant([0.2275], name='masscart')
            self.masspole = tf.constant([0.0923])
            self.friction_coef = tf.constant([0.1])
            self.joint_friction_coef =  tf.constant([0.01])
            self.total_mass = (self.masspole + self.masscart)

            # self.observe_noise_scale = np.array([0.005, 0.01, 0.02, 0.01])

            self.length = tf.constant([0.185])  # actually half the pole's length
            self.polemass_length = (self.masspole * self.length)
            # self.force_mag = 9.8 * 2.6 / 1.75
            self.max_duty_ratio = 4500
            self.tau = 0.01  # seconds between state updates

            self.predicted_s_ = self.dynamic(self.s, self.motor_force, self.x_friction, self.theta_friction, self.residual)
            p_x, p_x_dot, p_theta, p_theta_dot = tf.split(self.predicted_s_, 4, axis=1)
            x, x_dot, theta, theta_dot = tf.split(self.s_, 4, axis=1)
            self.loss = loss = tf.losses.mean_squared_error(100 *tf.concat([x_dot, theta_dot], axis=1), 100*tf.concat([p_x_dot, p_theta_dot], axis=1))
            self.train = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.eval:
            succseful_loaded = self.restore()
            if succseful_loaded:
                print('model loaded')
            else:
                print('model loading failed')
                return
        else:
            self.dataset_size = self.process_data()
            self.learn()

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

    def process_data(self):

        data_path = './data/data_'
        file_list = []
        for i in range(45):
            name = data_path + str(i) +'.npz'
            try:
                file_list.append(np.load(name))
            except:
                continue


        self.dataset['s'] = [file['s'][0:-1,:] for file in file_list]
        self.dataset['s_'] = [file['s'][1:, :] for file in file_list]
        self.dataset['a'] = [file['a'][0:-1,:]/7200 for file in file_list]
        for key in self.dataset.keys():
            self.dataset[key] = np.concatenate(self.dataset[key], axis=0)
        validation_indices = np.random.choice(len(self.dataset['s']), size=int(len(self.dataset['s'])*self.validation_data_capacity), replace=False)
        for key in self.dataset.keys():
            self.validation_dataset[key] = self.dataset[key][validation_indices]
            self.dataset[key] = np.delete(self.dataset[key], validation_indices, axis=0)
        return len(self.dataset[key])

    def learn(self):
        validation_interval = int(self.steps * self.validation_frequency)
        for i in range(self.steps):
            lr_now = self.lr * (self.steps-i)/self.steps
            indices = np.random.choice(self.dataset_size, size=self.batch_size, replace=False)
            feed_dict = {self.s: self.dataset['s'][indices],
                         self.s_: self.dataset['s_'][indices],
                         self.a: self.dataset['a'][indices],
                         self.lr_holder:lr_now}
            loss, _ = self.sess.run([self.loss, self.train], feed_dict)
            if i%validation_interval ==0:
                feed_dict = {self.s: self.validation_dataset['s'],
                             self.s_: self.validation_dataset['s_'],
                             self.a: self.validation_dataset['a']}
                validation_loss = self.sess.run(self.loss, feed_dict)
                string_to_print = ['time_step:', str(i), '|',
                                   'loss:', str(round(loss, 6)), '|'
                                   'validation_loss:', str(round(validation_loss, 6))]
                print(''.join(string_to_print))
        self.save_result()
        return

    def _build_motor_net(self, s, a, name='motor_force', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64  # 30
            w1_s = tf.get_variable('w1_s', [4, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.sigmoid(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 32, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_friction_net(self, s, a, name='friction', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64  # 30
            w1_s = tf.get_variable('w1_s', [4, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.sigmoid(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            x_friction = tf.layers.dense(net_1, 1, trainable=trainable)
            theta_friction = tf.layers.dense(net_1, 1, trainable=trainable)
            return x_friction, theta_friction

    def _build_residual_net(self, s, a, name='residual', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64  # 30
            w1_s = tf.get_variable('w1_s', [4, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.sigmoid(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 32, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 4, trainable=trainable)  # Q(s,a)

    def dynamic(self,s, F, x_frition, theta_friction, residual):
        x, x_dot, theta, theta_dot = tf.split(s, 4, axis= 1)

        # force = action
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)
        temp = (F - self.friction_coef * x_dot + self.polemass_length * tf.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - self.joint_friction_coef * theta_dot - costheta * temp) / (
                self.length * (tf.constant([1.], shape=[1,1]) - self.masspole * tf.square(costheta) / self.total_mass))

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc

        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        predicted_next_s = tf.concat([x, x_dot, theta, theta_dot], axis=1)
        return predicted_next_s

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observe(self, s):
        noise = np.random.normal(0., 1., 4) * self.observe_noise_scale
        observation = copy(s) + noise
        return observation

    def restore(self):
        model_file = tf.train.latest_checkpoint('./envs/data/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        return success_load

    def save_result(self):

        save_path = self.saver.save(self.sess, "./data/model.ckpt")
        print("Save to path: ", save_path)

    def step(self, action, impulse=0, process_noise=np.zeros([5])):
        a = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)/7200
        # self.gravity = np.random.normal(10, 2)
        # self.masscart = np.random.normal(1, 0.2)
        # self.masspole = np.random.normal(0.1, 0.02)

        feed_dict = {self.s:[self.state], self.a: [action]}
        s_ = self.sess.run(self.predicted_s_, feed_dict)[0]
        self.state = s_
        x, x_dot, theta, theta_dot = s_
        
        observation = self.observe(self.state)

        done = abs(x) > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians \
               or abs(x_dot) > self.max_v \

        done = bool(done)
        if x < -self.x_threshold \
                or x > self.x_threshold:
            a = 1

        cost = np.sum(self.cost_weight * np.array([x**2/self.x_threshold**2,
                                                   (theta/ self.theta_threshold_radians)**2,
                                                   abs(x_dot),
                                                   abs(theta_dot),
                                                   (action[0] / self.max_duty_ratio) ** 2]))


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

if __name__ == '__main__':
    env = CartPoleEnv_adv()

