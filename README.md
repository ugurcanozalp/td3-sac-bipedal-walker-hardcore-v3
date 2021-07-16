# Bipedal Walker Hardcore (and Classic) with SAC and TD3

Bipedal Walker environments of GYM are difficult problems to solve by reinforcement learning. 

In this repository, my thesis work is available. Various neural network architectures and RL methods implementations for solving BipedalWalker-v3 and BipedalWalkerHardcore-v3 of GYM on PyTorch using Soft Actor Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3). 

## Neural Nets
- [x] Feed Forward Neural Network 
- [x] Long Short Term Memory 
- [x] Transformer (pre-layer normalized)

Only Hardcore environment is solved by SAC and TD3 algorithm. Reward is manipulated and frame rate is halved. 

# How to
Create new python environment and First install requirements. (python 3.6)

```bash
pip install -r requirements.txt
```

## Training

Train your model via following commands.

Train Feed Forward NN with SAC
```bash
python main_script.py -f train -r sac -m mlp
```

Train LSTM (6 obs hist) with SAC
```bash
python main_script.py -f train -r sac -m lstm -hl 6
```

Train LSTM (12 obs hist) with SAC
```bash
python main_script.py -f train -r sac -m lstm -hl 12
```

Train Transformer (6 obs hist) with SAC
```bash
python main_script.py -f train -r sac -m trsf -hl 6
```

Train Transformer (12 obs hist) with SAC
```bash
python main_script.py -f train -r sac -m trsf -hl 12
```

-----------------------------------------------------------------------

Train Feed Forward NN with TD3
```bash
python main_script.py -f train -r td3 -m mlp
```

Train LSTM (6 obs hist) with TD3
```bash
python main_script.py -f train -r td3 -m lstm -hl 6
```

Train LSTM (12 obs hist) with TD3
```bash
python main_script.py -f train -r td3 -m lstm -hl 12
```

Train Transformer (6 obs hist) with TD3
```bash
python main_script.py -f train -r td3 -m trsf -hl 6
```

Train Transformer (12 obs hist) with TD3
```bash
python main_script.py -f train -r td3 -m trsf -hl 12
```

## Simulating pretrained models
Download pretrained models from the following link and place onto models folder
https://drive.google.com/drive/folders/1BtqZXrJyuoBiyeE9IduWj7IkFN-urw6y?usp=sharing

Then run one of the following commands,

```bash
python main_script.py -f test -m mlp -c ep4000
```

```bash
python main_script.py -f test -m trsf -h 12 -c ep4600
```

```bash
python main_script.py -f test -m lstm -h 12 -c ep4900
```

# Author

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ugurcanozalp/)

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@uurcann94)

[![StackOverFlow](https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/11985314/u%c4%9fur-can-%c3%96zalp)
