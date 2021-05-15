# Clustered Pruning for Higher Robustness to Gradient Leakage
## CS242 Final Project, Spring 2021
## Javin Pombra, Lucy Liu, Meiling Thompson

<img src = "https://github.com/lliu12/cpdefense/blob/master/cp_diagram.png" width="700" height="500" />

Reducing inter-device communication through pruning in federated learning settings increases robustness to data leakage attacks and decreases communication cost. However, pruning often results in decreases in model accuracy. Through our implementation of a clustered pruning method we develop, we show that selectively pruning by dividing the gradient among devices in a cluster preserves model accuracy while maintaining a high pruning percentage and preventing deep leakage through gradients.

============================================================

To test out our pruning experiments, use the Clustered_Pruning_IID.py and Clustered_Pruning_NonIID.py files.

You can alter any of the following command line arguments: 

python Clustered_Pruning_IID.py --initial-rounds 1 --initial-epochs 3 --num-devices 60 --fl-rounds 50 --fl-epochs 1 --device-percent 1.0 --data-percent 0.1 --num-clusters 10 --pca-dim 4

The same holds for NonIID, however instead of --num-devices , you can change --num-items-per-device

A demonstration of our privacy assessment code is in leakage.ipynb. 