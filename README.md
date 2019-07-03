# Reinforcement Learning with Robust Stability Guarantee

## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.


To create a conda env with python3, one runs 
```bash
conda create -n test python=3.6
```
To activate the env: 
```
conda activate test
```

# Installation Environment

```bash
git clone https://github.com/RLControlTheoreticGuarantee/Guarantee_Learning_Control
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib

```
### Example 1. RLAC with continous cartpole
```
python main.py
```
The hyperparameters, the tasks and the learning algorithm can be changed via change the variant.py, for example:


The algorithm_name could be one of ['RLAC','RARL','SAC_cost']


Other hyperparameter are also ajustable in variant.py.
```bash
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
```
## Reference

[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

[2] [Baselines](https://github.com/openai/baselines)

[3] [Rllab](https://github.com/rll/rllab)
