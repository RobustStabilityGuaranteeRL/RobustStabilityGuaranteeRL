from collections import OrderedDict, deque
import numpy as np
from copy import deepcopy

class Pool(object):

    def __init__(self, variant):

        s_dim = variant['s_dim']
        a_dim = variant['a_dim']
        d_dim = variant['d_dim']
        self.memory_capacity = variant['memory_capacity']
        store_last_n_paths = variant['store_last_n_paths']
        self.paths = deque(maxlen=store_last_n_paths)
        self.reset()
        self.memory = {
            's': np.zeros([1, s_dim]),
            'a': np.zeros([1, a_dim]),
            'd': np.zeros([1, d_dim]),
            'raw_d': np.zeros([1, d_dim]),
            'r': np.zeros([1, 1]),
            'terminal': np.zeros([1, 1]),
            's_': np.zeros([1, s_dim]),

        }
        if variant['value_horizon'] is not None:
            self.memory.update({'value':np.zeros([1, 1])}),
            self.horizon = variant['value_horizon']
        self.memory_pointer = 0
        self.min_memory_size = variant['min_memory_size']

    def reset(self):
        self.current_path = {
            's': [],
            'a': [],
            'd': [],
            'raw_d':[],
            'r': [],
            'terminal': [],
            's_': [],
        }

    def store(self, s, a, d, raw_d, r, terminal, s_):
        transition = {'s': s, 'a': a, 'd': d,'raw_d':raw_d, 'r': np.array([r]), 'terminal': np.array([terminal]), 's_': s_}
        if len(self.current_path['s']) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis,:]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate((self.current_path[key],transition[key][np.newaxis,:]))

        if terminal == 1.:
            if 'value' in self.memory.keys():
                r = deepcopy(self.current_path['r'])
                path_length = len(r)
                last_r = self.current_path['r'][-1, 0]
                r = np.concatenate((r,last_r*np.ones([self.horizon,1])), axis=0)
                value = []
                [value.append(r[i:i+self.horizon,0].sum()) for i in range(path_length)]
                value = np.array(value)
                self.memory['value'] = np.concatenate((self.memory['value'], value[:, np.newaxis]), axis=0)
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate((self.memory[key], self.current_path[key]), axis=0)
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory['s'])

        return self.memory_pointer

    def sample(self, batch_size):
        if self.memory_pointer < self.min_memory_size:
            return None
        else:
            indices = np.random.choice(min(self.memory_pointer,self.memory_capacity)-1, size=batch_size) \
                      + max(1, 1+self.memory_pointer-self.memory_capacity)*np.ones([batch_size],np.int)
            batch = {}
            [batch.update({key: self.memory[key][indices]}) for key in self.memory.keys()]

            return batch


