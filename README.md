# LSM-Simulator

## Requirements
python 3.7

Install package:
```
pip install -r requirements.txt 
```

## Run the simulation
```
python main.py
```

## Datasets
Two types of dataset are provided in this simulator.

TI46: To use TI46, change the variable data_set in main.py. The dataset size can be chosen from speaker_per_class.

MNIST: To use MNIST, change the variable data_set in main.py.

## Networks
The network is built up in network.py. Any number of layers can be set. The layers are connected according to inputs/outputs matrices.
