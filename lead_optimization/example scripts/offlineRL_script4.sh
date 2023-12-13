sleep 900
export MAIN_DIR=/home/abhor/ReactionRL

m=actor-critic
n=random
al=PG

for seed in 0 1 2 3 4
    do
    for s in 1 2 5
    do
        python offlineRL.py --steps $s --model $m --actor-loss $al --neg $n --cuda 3 --seed $seed
    done
done