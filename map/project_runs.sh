##### training 

# Cooperative
# -----------
# Coop nav. (simple_spead_in)
# Hetero nav. (hetero_spread_in)

# Competitive
# -----------
# Phy decep (simple_adversary_in)
# Pred-prey (simple_tag_in)
# Keep-away (simple_push_in)

##### -- Cooperative -- ##### 

# Scenario 1
# Coop Nav (5v0) 
# elign_team
CUDA_VISIBLE_DEVICES=0 python train_multi_sacd.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_spread_elign_team
CUDA_VISIBLE_DEVICES=1 python train_multi_sacd_cond.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_spread_cond_elign_team

# Scenario 2
# Hetero Nav (6v0) 
# elign_team
CUDA_VISIBLE_DEVICES=2 python train_multi_sacd.py --task hetero_spread_in --num-good-agents 6 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/hetero_spread_elign_team
CUDA_VISIBLE_DEVICES=3 python train_multi_sacd_cond.py --task hetero_spread_in --num-good-agents 6 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/hetero_spread_cond_elign_team

##### -- Competition -- ##### 

# Scenario 1
# Phy-decep (4v2) (simple_adversary_in)
# elign_team
CUDA_VISIBLE_DEVICES=4 python train_multi_sacd.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_adversary_elign_team
CUDA_VISIBLE_DEVICES=5 python train_multi_sacd_cond.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_adversary_cond_elign_team
# elign_adv
CUDA_VISIBLE_DEVICES=6 python train_multi_sacd.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_adversary_elign_adv
CUDA_VISIBLE_DEVICES=7 python train_multi_sacd_cond.py --task simple_adversary_in --num-good-agents 4 --num-adversaries 2 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_adversary_cond_elign_adv

# Scenario 2
# Pred-prey (4v4) (simple_tag_in)
# elign_team
CUDA_VISIBLE_DEVICES=0 python train_multi_sacd.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_tag_elign_team
CUDA_VISIBLE_DEVICES=1 python train_multi_sacd_cond.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_tag_cond_elign_team
# elign_adv
CUDA_VISIBLE_DEVICES=2 python train_multi_sacd.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_tag_elign_adv
CUDA_VISIBLE_DEVICES=3 python train_multi_sacd_cond.py --task simple_tag_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_tag_cond_elign_adv

# Scenario 2
# Keep-away (4v4) (simple_push_in)
# elign_team
CUDA_VISIBLE_DEVICES=4 python train_multi_sacd.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_push_elign_team
CUDA_VISIBLE_DEVICES=5 python train_multi_sacd_cond.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_push_cond_elign_team
# elign_adv
CUDA_VISIBLE_DEVICES=6 python train_multi_sacd.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_push_elign_adv
CUDA_VISIBLE_DEVICES=7 python train_multi_sacd_cond.py --task simple_push_in --num-good-agents 4 --num-adversaries 4 --obs-radius 0.5 --intr-rew elign_adv --epoch 50 --save-models --benchmark  --logdir log/simple_push_cond_elign_adv



## Conditioning on behavior
CUDA_VISIBLE_DEVICES=8 python train_multi_sacd_embed_cond.py --task simple_spread_in --num-good-agents 5 --obs-radius 0.5 --intr-rew elign_team --epoch 50 --save-models --benchmark  --logdir log/simple_spread_embed_cond_elign_team