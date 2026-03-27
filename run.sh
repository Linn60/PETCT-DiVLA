torchrun --nproc_per_node=8 --master_port=29602 src/train/train_petct_CLIP.py \
    --root_dir /data/home/run \
    --language_model_name_or_path models/mmBERT/mmBERT-small/ \
    --output_dir output/ \
    \
    --img_size 192 192 336 \
    --patch_size 16 16 12 \
    --dim 384 \
    --mlp_dim 1536 \
    --depth 12 \
    --heads 6 \
    \
    --local_loss False \
    --gather_loss True \
    --bf16 True \
    --stage2_step 0 \
    --stage1_lambda_loc 0.1 \
    --lambda_loc 0.1 \
    --lambda_ent 0.1 \
    --lambda_pr 0.1 \
    \
    --data_root /data/home/run/data \
    --image_root PETCT/images \
    --reports_path PETCT/reports.xlsx \
    --test_size 800 \
    --labels_path PETCT/aggregated_labels.xlsx \
    --organ_labels_path PETCT/organ_labels.jsonl \
    --seg_root PETCT/seg \
    --preload False \
    \
    --max_steps 10000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 7 \
    --logging_steps 10 \
    \
    --eval_retrieval_steps 250 \
    --eval_retrieval_samples 512 \
    --eval_retrieval_batch_size 32 \
    \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 6 \

