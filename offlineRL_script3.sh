sleep 600

export MAIN_DIR=/home/abhor/ReactionRL

m=actor-critic
n=random
al=mse

for s in 1 2 5
do
    python offlineRL.py --steps $s --model $m --actor-loss $al --neg $n --cuda 2
done
