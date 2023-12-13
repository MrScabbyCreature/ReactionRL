# ReactionRL
This is the code for \<paper\>. Reaction rules are extracted from USPTO-MIT dataset and used as actions for reinforcement learning for molecular applications. Two applications are created:  
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
networkx
torchdrug
````

#### Requirements for drug discovery
```
gymnasium
tensorboard
stable_baselines3
```


#### Requirements for lead optimization
```
tabulate
matplotlib
```

# Usage
#### Extract action dataset from USPTO-MIT
```
./preprocess.sh
```

#### Molecular Discovery
The gymnasium environment for molecular discovery is contained in folder `molecular_discovery`. Folder `sb3` contains 4 example \<agents\> = [ppo/sac/td3/ddpg]. For molecular discovery using stable_baselines3, run 

```
python -m sb3.<agent> --timesteps 1000000 mode train --reward [logp\qed\drd2\SA]
```

#### Lead optimization
```
# generate some offline data by rolling out a random policy.
python lead_optimization.dump_data_for_offlineRL --train 100000 --steps 5

# train an offline RL agent
python lead_optimization.python offlineRL.py --steps 5 --model actor-critic --actor-loss PG --cuda 0 
```

