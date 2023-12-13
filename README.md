# ReactionRL
This is the code for <paper>. Reaction rules are extracted from USPTO-MIT dataset and used as actions for reinforcement learning for molecular applications. Two applications are created:  
1. Drug discovery using a gymnasium-compatible RL simulator (Online RL)
2. Lead optimization without using similarity-based metrics (Goal-conditioned RL + offline RL)


# Requirements
This repo was built using python=3.7.  
Common requirements:
```
deepchem
notebook
pandas
RDKit
filehash
pytorch
````

#### Requirements for drug discovery
```
gymnasium
tensorboard
stable_baselines3
```


#### Requirements for lead optimization
```
torchdrug
tabulate
matplotlib
```

# Usage
1. Download and process USPTO-MIT dataset: `./preprocess.sh` 
2. 
