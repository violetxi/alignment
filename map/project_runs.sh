CUDA_VISIBLE_DEVICES=5 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s0 --seed 0
CUDA_VISIBLE_DEVICES=6 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s1 --seed 1
CUDA_VISIBLE_DEVICES=7 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s0 --seed 0

# run it with new intrinsic motivations
CUDA_VISIBLE_DEVICES=3 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew new_im --epoch 50 --save-models --benchmark --logdir log/simple_spread_cond_elign_team