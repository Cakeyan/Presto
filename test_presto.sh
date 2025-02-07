export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CHECKPOINT="10000"
MODEL="./checkpoint-"$CHECKPOINT"/model"

GS=7.0
EVAL_FILE="./resources/t2v_test_67_multi_5.jsonl"

echo $MODEL
echo $CHECKPOINT
echo $GS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 \
    file_inference.py \
    --user_file $EVAL_FILE \
    --file_save_root ./output_videos/$CHECKPOINT"_"$GS \
    --tokenizer /huggingface/rhymes-ai/Allegro/tokenizer \
    --text_encoder /huggingface/rhymes-ai/Allegro/text_encoder \
    --vae /huggingface/rhymes-ai/Allegro/vae \
    --vae_load_mode decoder_only \
    --dit $MODEL \
    --guidance_scale $GS \
    --num_sampling_steps 100 \
    --enable_vae_compile \
    --seed 1
