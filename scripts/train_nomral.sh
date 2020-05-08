for nlev in 100 200; do
    for seed in try1 try2 try3; do
        python -m train_procgen.train --env_name $1 --num_levels $nlev --exp_name $seed 
    done
done
