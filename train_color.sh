CUDA_VISIBLE_DEVICES=1 python training/train.py \
--input_nc=1 \
--ref_input_nc=3 \
--dataset_type=cloth_encode \
--exp_dir=/data/shijianyang/code/pixel2style2pixel/path/to/experiment_sketch2d \
--workers=8 \
--output_size=256 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--moco_lambda=0.1 \
--adv_lambda=1 \
--use_wandb