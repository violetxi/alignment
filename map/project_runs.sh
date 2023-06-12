CUDA_VISIBLE_DEVICES=5 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s0 --seed 0
CUDA_VISIBLE_DEVICES=6 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s1 --seed 1
CUDA_VISIBLE_DEVICES=7 python train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 100 --save-models --benchmark  --logdir log/simple_spread_elign_team_s0 --seed 0

# run it with new intrinsic motivations
# coorperation game
CUDA_VISIBLE_DEVICES=0 python map/train_multi_sacd_new_im.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew new_im_both --epoch 100 --save-models --benchmark --logdir log/simple_spread_cond_new_im_both_s_0 --seed 0


CUDA_VISIBLE_DEVICES=1 python map/train_multi_sacd_new_im.py --task hetero_spread_in --num-good-agents 6 --obs-radius 0.5 --intr-rew new_im_both --epoch 100 --save-models --benchmark --logdir log/hetero_spread_cond_new_im_both_s_0 --seed 0

# adversary game
# Scenario 1: physical deception
CUDA_VISIBLE_DEVICES=2 python map/train_multi_sacd_new_im.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew new_im_both --epoch 100 --save-models --benchmark --logdir log/simple_adversary_cond_new_im_both_s_0 --seed 0
CUDA_VISIBLE_DEVICES=7 python map/train_multi_sacd_new_im.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew new_im_team --epoch 50 --save-models --benchmark --logdir log/simple_adversary_cond_new_im_team_s_0 --seed 0
CUDA_VISIBLE_DEVICES=5 python map/train_multi_sacd_new_im.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew new_im_adv --epoch 50 --save-models --benchmark --logdir log/simple_adversary_cond_new_im_adv_s_0 --seed 0
# Scenario 2: Pred-prey (4 v 4)
CUDA_VISIBLE_DEVICES=6 python map/train_multi_sacd_new_im.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_both --epoch 50 --save-models --benchmark --logdir log/simple_tag_in_cond_new_im_both_s_0 --seed 0
CUDA_VISIBLE_DEVICES=0 python map/train_multi_sacd_new_im.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_team --epoch 50 --save-models --benchmark --logdir log/simple_tag_in_cond_new_im_team_s_0 --seed 0
CUDA_VISIBLE_DEVICES=1 python map/train_multi_sacd_new_im.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_adv --epoch 50 --save-models --benchmark --logdir log/simple_tag_in_cond_new_im_adv_s_0 --seed 0
# Scenario 3: Keep away
CUDA_VISIBLE_DEVICES=2 python map/train_multi_sacd_new_im.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_both --epoch 50 --save-models --benchmark --logdir log/simple_push_in_cond_new_im_both_s_0 --seed 0
CUDA_VISIBLE_DEVICES=0 python map/train_multi_sacd_new_im.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_team --epoch 50 --save-models --benchmark --logdir log/simple_push_in_cond_new_im_team_s_0 --seed 0
CUDA_VISIBLE_DEVICES=1 python map/train_multi_sacd_new_im.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew new_im_adv --epoch 50 --save-models --benchmark --logdir log/simple_push_in_cond_new_im_adv_s_0 --seed 0