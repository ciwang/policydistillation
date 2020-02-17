# cs234-policydist

## NOTE: THIS CODE IS NOT MAINTAINED. PLEASE DO NOT USE IT.

Reproducing the algorithm described in [Rusu et al., 2016](https://arxiv.org/abs/1511.06295). 

"Quick" start:
- Run ```python natureqn_atari.py``` to train the teacher netowork. (This will take ~12 hours.) Skip this step if trained Tensorflow DQN for Pong is saved as a checkpoint.
- Run ```python distilledqn_atari.py``` to train the student network. Make sure the loss function and checkpoint directory are correct.
