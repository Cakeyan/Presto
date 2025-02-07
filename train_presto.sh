export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_API_KEY=YOUR_KEY

accelerate launch \
    --num_machines $WORLD_SIZE \
    --num_processes $(($NPROC_PER_NODE * $WORLD_SIZE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --config_file config/accelerate_config.yaml \
    train_presto.py \
    --project_name Presto_88x720p \
    --dit_config /huggingface/rhymes-ai/Allegro/transformer/config.json \
    --dit /huggingface/rhymes-ai/Allegro/transformer/ \
    --tokenizer /huggingface/rhymes-ai/Allegro/tokenizer \
    --text_encoder /huggingface/rhymes-ai/Allegro/text_encoder \
    --vae /huggingface/rhymes-ai/Allegro/vae \
    --vae_load_mode encoder_only \
    --enable_vae_compile \
    --dataset presto \
    --data_dir /cpfs/data/user/ \
    --meta_file ./resources/train_data_template.parquet \
    --num_frames 88 \
    --max_height 720 \
    --max_width 1280 \
    --hw_thr 1.0 \
    --hw_aspect_thr 1.5 \
    --dataloader_num_workers 10 \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000000 \
    --learning_rate 1e-4 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --mixed_precision bf16 \
    --report_to wandb \
    --allow_tf32 \
    --enable_stable_fp32 \
    --model_max_length 512 \
    --cfg 0.1 \
    --checkpointing_steps 50 \
    --resume_from_checkpoint latest \
    --output_dir ./output/Presto_88x720p \
    --num_prompts 5 \
    --stride_per_prompt 90 \
    