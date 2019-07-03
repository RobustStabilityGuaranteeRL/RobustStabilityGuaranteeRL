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
### Example 1. 
```
python main_for_sac.py
```
The hyperparameters, the tasks and the learning algorithm can be changed via change the variant.py, for example:



The env_name could be one of ['CartPolecons-v0','CartPolecost-v0','Antcons-v0', 'HalfCheetahcons-v0','Pointcircle-v0','Quadrotorcons-v0','Quadrotorcost-v0','FetchReach-v1', 'Carcost-v0']




The algorithm_name could be one of ['SAC_lyapunov', 'SAC', 'SSAC','CPO', 'CPO_lyapunov', 'PDO', 'DDPG','LAC','SAC_cost']



Other hyperparameter are also ajustable in variant.py.
```bash
VARIANT = {
    'env_name': 'CartPolecons-v0',
    'algorithm_name': 'SAC_lyapunov',
    'additional_description': '-Test',
    'evaluate': False,
    'train':True,
    'evaluation_frequency': 2048,
    'num_of_paths': 1,
    'num_of_trials': 5,
    'store_last_n_paths': 10,
    'start_of_trial': 0,
}
```
## Reference

[1] [Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

[2] [Baselines](https://github.com/openai/baselines)

[3] [Rllab](https://github.com/rll/rllab)
