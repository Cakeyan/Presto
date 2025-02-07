import torch
import torch.distributed as dist
import imageio
import os
import argparse
import jsonlines
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
from prompts import POS_INFERENCE_PROMPT, NEG_INFERENCE_PROMPT

def single_inference(pipeline, input_text, output_path, args):
    out_video = pipeline(
        input_text, 
        negative_prompt = NEG_INFERENCE_PROMPT, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=512,
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    ).video[0]

    imageio.mimwrite(output_path, out_video, fps=15, quality=8)  # highest quality is 10, lowest is 0
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_prompt", type=str, default='')
    parser.add_argument("--user_file", type=str, default='')
    parser.add_argument("--vae", type=str, default='')
    parser.add_argument("--vae_load_mode", type=str, default='decoder_only')
    parser.add_argument("--dit", type=str, default='')
    parser.add_argument("--text_encoder", type=str, default='')
    parser.add_argument("--tokenizer", type=str, default='')
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--file_save_root", type=str, default="./output_videos/")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cpu_offload", action='store_true')
    parser.add_argument("--enable_vae_compile", action='store_true')
    parser.add_argument("--enable_dit_compile", action='store_true')

    args = parser.parse_args()

    if os.path.dirname(args.save_path) != '' and (not os.path.exists(os.path.dirname(args.save_path))):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    if not os.path.exists(args.file_save_root):
        os.makedirs(args.file_save_root, exist_ok=True)

    # multi-gpu setup
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        
    # pipeline configuration    
    dtype=torch.bfloat16
    device = torch.cuda.current_device()

    # vae have better formance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32, load_mode=args.vae_load_mode).to(device=device)
    vae.eval()
    if args.enable_vae_compile:
        vae.decoder = torch.compile(vae.decoder, mode="max-autotune", fullgraph=True)
        print("vae compiled")
    print("it's normal that vae is not full loaded in decoder-only mode")

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device=device)
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer,
    )

    scheduler = EulerAncestralDiscreteScheduler()

    transformer = AllegroTransformer3DModel.from_pretrained(
        args.dit,
        torch_dtype=dtype
    ).to(device=device)
    transformer.eval()

    allegro_pipeline = AllegroPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    ).to(device=device)

    if args.enable_dit_compile:
        allegro_pipeline.transformer = torch.compile(allegro_pipeline.transformer)
        print("transformer compiled")

    if args.enable_cpu_offload:
        allegro_pipeline.enable_sequential_cpu_offload()
        print("cpu offload enabled")

    if args.user_file:
        data_list = []
        with jsonlines.open(args.user_file, 'r') as reader:
            for data in reader:
                data_list.append(data)
        if len(data_list) % world_size != 0:
            data_list = data_list + data_list[:world_size - len(data_list) % world_size]
        
        text_path_list = []
        for data in data_list:
            user_prompt = [POS_INFERENCE_PROMPT.format(p.lower().strip()) for p in data['cap']]
            save_path = os.path.join(args.file_save_root, f"{data['name']}.mp4")
            text_path_list.append((user_prompt, save_path))

    elif args.user_prompt:
        user_prompt = POS_INFERENCE_PROMPT.format(args.user_prompt.lower().strip())
        text_path_list = [(user_prompt, args.save_path)]

    for i, (input_text, output_path) in enumerate(text_path_list):
        if i % world_size == rank:
            single_inference(allegro_pipeline, input_text, output_path, args)
            