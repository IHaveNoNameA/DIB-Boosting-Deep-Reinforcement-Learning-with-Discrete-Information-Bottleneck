CUDA_VISIBLE_DEVICES=1 python train.py \
    --domain_name finger \
    --task_name spin \
    --encoder_type pixel \
    --action_repeat 2 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./data/finger \
    --agent curl_sac --frame_stack 3 \
    --max_iter 1  \
    --seed -1 --critic_lr 1e-4 --actor_lr 1e-3 --eval_freq 10000 --batch_size 512 --num_train_steps 500000