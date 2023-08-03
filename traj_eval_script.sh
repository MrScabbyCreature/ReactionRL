N=1000
cuda=0

# Run for each offlineRL model
for train_steps in 1 2 5; do 
    model=models/supervised/offlineRL/emb_model_update\=1\|\|steps\=$train_steps\|\|topk\=10/model.pth 

    # Run with branch=[5, 10] 
    for br in 5 10; do

        # for eval steps = 3
        for ls in 1 2; do
            python trajectory_eval_for_offlineRL.py --model-path $model --cuda $cuda --load-step $ls --N $N --branch $br --eval-for-steps 3

            for ts in 1 5 10 20 50 100; do
                python trajectory_eval_for_offlineRL.py --model-path $model --cuda $cuda --load-step $ls --N $N --branch $br --eval-for-steps 3 --top-sim $ts
            done

        done
        # for eval steps = 10
        for ls in 5 10; do
            for ts in 1 5 10 20 50 100; do
                python trajectory_eval_for_offlineRL.py --model-path $model --cuda $cuda --load-step $ls --N $N --branch $br --eval-for-steps 10 --top-sim $ts
            done
        done
    done
done
