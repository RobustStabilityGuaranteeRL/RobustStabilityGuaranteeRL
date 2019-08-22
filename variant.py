import gym
import datetime
import numpy as np
import ENV.env
SEED = None
ITA = 1.
VARIANT = {
    # 'env_name': 'cartpole_cost_with_fitted_motor',
    # 'env_name': 'HalfCheetahcost-v0',
    'env_name': 'cartpole_cost',
    #training prams
    'algorithm_name': 'RLAC',
    # 'algorithm_name': 'RARL',
    # 'algorithm_name': 'SAC_cost',
    'disturber': 'SAC',
    # 'additional_description': '-horizon=inf-weight-4-10-0-0-0',
    # 'additional_description': '-new',
    'additional_description': '-horizon=20-256-256',
    # 'evaluate': False,
    'train': True,
    # 'train': False,

    'num_of_trials': 5,   # number of random seeds
    'store_last_n_paths': 10,  # number of trajectories for evaluation during training
    'start_of_trial': 5,

    #evaluation params
    # 'evaluation_form': 'constant_impulse',
    # 'evaluation_form': 'impulse',
    'evaluation_form': 'various_disturbance',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [
        # 'RLAC',
        'RLAC-video',
        # 'RLAC-LAC-horizon=inf',
        # 'RLAC-horizon=inf-dis=.1',
        # 'RLAC-horizon=inf-in_turn_train-256-256',
        # 'SAC_cost',
        # 'RLAC-horizon=20-256-256-inturn',
        # 'RLAC-finite_horizon-15',
        # 'RLAC-horizon=inf-256-256',
        # 'RLAC-finite_horizon-20',
        # 'RLAC-finite_horizon-5',
        # 'RLAC-LAC-non-finite',
        # 'MPC',
        # 'RLAC-LAC',
        # 'RLAC-LAC-horizon=10-weight-1-10-0-0-0',
        # 'RLAC-LAC-history=5',
        # 'SAC_cost',
        # 'RLAC-LAC-infinite-horizon',
        # 'RLAC-horizon=50-dis=.1',
        # 'RLAC-LAC',
        # 'LQR',
        # 'MPC',
        # 'RARL',
        # 'RLAC-finite_horizon',
        # 'RARL-new',
        # 'SAC',
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
        'eval_render': True,},
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': False,},
    'cartpole_cost_v2': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},

    'cartpole_cost_partial': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},

    'cartpole_cost_real': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
    'cartpole_cost_real_no_friction': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
    'cartpole_cost_swing_up': {
        'max_ep_steps': 5000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
    'cartpole_cost_with_motor': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,},
    'cartpole_cost_with_fitted_motor': {
        'max_ep_steps': 1000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': True,},
}
ALG_PARAMS = {
    'MPC':{
        'horizon': 5,
    },

    'LQR':{
        'use_Kalman': False,
    },

    'RLAC': {
        'iter_of_actor_train_per_epoch': 50,
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
        'value_horizon': 20,
        'finite_horizon': True,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
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
        'steps_per_cycle': 100,
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
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'start_of_disturbance':0,
        # 'start_of_disturbance': 0,
        'adaptive_alpha': True,
        'target_entropy': None,
        'energy_bounded': False,
        # 'energy_bounded': True,
        # 'process_noise': True,
        'process_noise': False,
        # 'noise_dim': 2,
        'energy_decay_rate': 0.5,
        # 'disturbance_magnitude': np.array([1]),
        # 'disturbance_magnitude': np.array([5, 0, 0, 0, 0]),
        'disturbance_magnitude': np.array([.1, .1, .1, .1, .1, .1]),
        # 'disturbance_magnitude': np.array([0.1, 1, 5,10]),
    },
}

EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(80, 155, 10),
        # 'magnitude_range': np.arange(0.5, 1., .1),
        'num_of_paths': 100,   # number of path for evaluation
        'impulse_instant': 100,
    },
    'constant_impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'magnitude_range': np.arange(80, 155, 5),
        'magnitude_range': np.arange(80, 155, 10),
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(.2, 3.2, .2),
        'num_of_paths': 500,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 10, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([80]),
        # 'grid_eval': False,
        'num_of_paths': 100,   # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        # 'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'path': './log/HalfCheetahcost-v0/RLAC-horizon=inf-dis=.1/0/',
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
    elif name == 'cartpole_cost_partial':
        from envs.ENV_V3 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real':
        from envs.ENV_V4 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_swing_up':
        from envs.ENV_V5 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_real_no_friction':
        from envs.ENV_V6 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_motor':
        from envs.ENV_V7 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost_with_fitted_motor':
        from envs.ENV_V8 import CartPoleEnv_adv as dreamer
        env = dreamer(eval=True)
        env = env.unwrapped
    elif name == 'Quadrotorcost-v0':
        env = gym.make('Quadrotorcons-v0')
        env = env.unwrapped
        env.modify_action_scale = False
        env.use_cost = True

    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotorcons-v0':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    env.seed(SEED)
    return env

def get_train(name):
    if 'RARL' in name:
        from LAC.RARL import train as train
    elif 'LAC' in name:
        from LAC.LAC_V1 import train
        # from LAC.LAC_V1 import train_v2 as train
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
    elif 'MPC' in name:
        from LAC.MPC import MPC as build_func
    else:
        from LAC.SAC_cost import SAC_cost as build_func
    return build_func

def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval


