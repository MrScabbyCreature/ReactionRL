sleep 300

export MAIN_DIR=/home/abhor/ReactionRL

m=actor-critic
n=closest
al=PG


for s in 1 2 5
do
    python offlineRL.py --steps $s --model $m --actor-loss $al --neg $n --cuda 1
done
