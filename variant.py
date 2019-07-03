import gym
import datetime
import numpy as np
SEED = None
ITA = 1
VARIANT = {
    'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'RLAC',
    # 'algorithm_name': 'RARL',
    # 'algorithm_name': 'SAC_cost',
    'disturber': 'SAC',
    'additional_description': '',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 10,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 0,

    #evaluation params
    'evaluation_form': 'impulse',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # 'RLAC',

    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 2048,
}
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
     }
ALG_PARAMS = {

    'RLAC': {
        'iter_of_actor_train_per_epoch': 150,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 1.,
        'ita': ITA,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 10,
        'target_entropy': None
    },
    'RARL': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'ita': 0,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 5,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None
    },

    'SAC_cost': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'ita': ITA,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': -5,

    },
}


DISTURBER_PARAMS = {

    'SAC': {
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'alpha': 1.,
        'ita': ITA,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 200,
        'train_per_cycle': 50,
        'adaptive_alpha': True,
        'target_entropy': None,
        'energy_bounded': False,
        # 'energy_bounded': True,
        # 'process_noise': True,
        'process_noise': False,
        # 'noise_dim': 2,
        'energy_decay_rate': 0.5,
        # 'disturbance_magnitude': np.array([1]),
        'disturbance_magnitude': np.array([5, 0, 0, 0, 0]),
        # 'disturbance_magnitude': np.array([0.1, 1, 5,10]),
    },
}

EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.2, 2.2, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.4, 2.2, 0.2),    # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'magnitude_range': np.arange(80, 125, 5),
        'num_of_paths': 500,   # number of path for evaluation

    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'num_of_paths': 100,   # number of path for evaluation

    }
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]
VARIANT['disturber_params']=DISTURBER_PARAMS[VARIANT['disturber']]
RENDER = True
def get_env_from_name(name):
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_v2':
        from envs.ENV_V2 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    env.seed(SEED)
    return env

def get_train(name):
    if 'RARL' in name:
        from LAC.RARL import train as train
    elif 'LAC' in name:
        from LAC.LAC_V1 import train
    else:
        from LAC.SAC_cost import train

    return train

def get_policy(name):
    if 'RARL' in name:
        from LAC.RARL import RARL as build_func
    elif 'LAC' in name :
        from LAC.LAC_V1 import LAC as build_func
    elif 'LQR' in name:
        from LAC.lqr import LQR as build_func
    else:
        from LAC.SAC_cost import SAC_cost as build_func
    return build_func

def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval


