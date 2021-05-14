# Clustered Pruning for Higher Robustness to Gradient Leakage
## CS242 Final Project, Spring 2021


To test out pruning experiments, use the Clustered_Pruning_IID.py and Clustered_Pruning_NonIID.py files.

You can alter any of the following command line arguments: 

python Clustered_Pruning_IID.py --initial-rounds 1 --initial-epochs 3 --num-devices 60 --fl-rounds 50 --fl-epochs 1 --device-percent 1.0 --data-percent 0.1 --num-clusters 10 --pca-dim 4

The same holds for NonIID, however instead of --num-devices , you can change --num-items-per-device
