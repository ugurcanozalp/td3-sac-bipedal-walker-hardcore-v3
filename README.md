# Bipedal Walker Hardcore (and Classic) with TD3

Bipedal Walker environments of GYM are difficult problems to solve by reinforcement learning. 

In this repository, my thesis work is available. Various neural network architectures and RL methods implementations for solving BipedalWalker-v3 and BipedalWalkerHardcore-v3 of GYM on PyTorch using Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3). 

## Neural Nets
- [x] Feed Forward Neural Network with Residual connection
- [x] Long Short Term Memory (4 last observations)
- [x] Bidirectional Long Short Term Memory (4 last observations)
- [x] Transformer (4 last observations)

Only Hardcore environment is solved by TD3 algorithm. Reward is manipulated and frame rate is halved. 

![FeedForward](results/video/ff-slow.gif)

![Transformer](results/video/trsf-slow.gif)

# How to
Create new python environment and First install requirements. (python 3.6)

```bash
pip install -r requirements.txt
```

## Training

Train your model via following commands. Change ff to lstm bilstm or trsf to train by other neural networks.

```bash
python main_script -f train -m ff
```

## Simulating pretrained models
Download pretrained models from the following link and place onto models folder
https://drive.google.com/drive/folders/1BtqZXrJyuoBiyeE9IduWj7IkFN-urw6y?usp=sharing

Then run one of the following commands,

```bash
python main_script -f test -m ff -c ep4000
```

```bash
python main_script -f test -m trsf -c ep4600
```

```bash
python main_script -f test -m lstm -c ep4900
```

# Author

