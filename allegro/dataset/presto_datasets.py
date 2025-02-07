import os
import pickle
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
import hashlib
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from accelerate.logging import get_logger

from allegro.utils.utils import text_preprocessing, lprint

logger = get_logger(__name__)

def filter_resolution(height, width, max_height, max_width, hw_thr, hw_aspect_thr):
    aspect = max_height / max_width
    if height >= max_height * hw_thr and width >= max_width * hw_thr and height / width >= aspect / hw_aspect_thr and height / width <= aspect * hw_aspect_thr:
        return True
    return False

def filter_duration_presto(num_frames, num_prompt, stride_per_prompt):
    target_frames = num_prompt * stride_per_prompt
    if num_frames >= target_frames:
        return True
    return False

class Presto_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.data_dir = args.data_dir
        self.meta_file = args.meta_file
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.hw_thr = args.hw_thr
        self.hw_aspect_thr = args.hw_aspect_thr
        self.cache_dir = args.cache_dir

        # multiple prompts
        self.num_prompts = args.num_prompts
        self.stride_per_prompt = args.stride_per_prompt

        self.filter_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data = self.data_list.loc[idx]
            if data['path'].endswith('.mp4'):
                return self.get_video(data)
            else:
                raise ValueError(f'Unsupported file type: {data["path"]}')
        except Exception as e:
            logger.info(f"Error with {e}, file {data['path']}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def get_video(self, data):
        vr = VideoReader(os.path.join(self.data_dir, data['path']))
        fidx = np.linspace(0, self.stride_per_prompt*self.num_prompts-1, self.num_frames, dtype=int, endpoint=True)
        sidx, eidx = self.temporal_sample(len(fidx))
        fidx = fidx[sidx: eidx]
        if self.num_frames != len(fidx):
            raise ValueError(f'num_frames ({self.num_frames}) is not equal with frame_indices ({len(fidx)})')
        video = vr.get_batch(fidx).asnumpy()
        video = torch.from_numpy(video)
        video = video.permute(0, 3, 1, 2)
        video = self.transform(video)
        video = video.transpose(0, 1)

        # sort and assertion
        if isinstance(data['cap'], dict):
            key, text = zip(*sorted(data['cap'].items(), key=lambda x: int(x[0])))
            text = list(text)
            assert int(key[1])-int(key[0]) == self.stride_per_prompt, f'The data meta dict of multiple prompts should keep the same with the stride_per_prompt arg, but {key[1]} - {key[0]} != {self.stride_per_prompt}'
        elif isinstance(data['cap'], str):
            text = [data['cap']]

        text = text_preprocessing(text)
        text = text[:self.num_prompts]
        for i in range(len(text)):
            if random.random() < self.cfg:
                text[i] = ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask)

    def filter_data_list(self):
        lprint(f'Filter data {self.meta_file}')
        cache_path = self.check_cache()
        if os.path.exists(cache_path):
            lprint(f'Load cache {cache_path}')
            with open(cache_path, 'rb') as f:
                self.data_list = pickle.load(f)
            lprint(f'Data length: {len(self.data_list)}')
            return
        
        self.data_list = pd.read_parquet(self.meta_file)
        pick_list = []
        for i in range(len(self.data_list)):
            data = self.data_list.loc[i]
            is_pick = filter_resolution(data['height'], data['width'], self.max_height, self.max_width, self.hw_thr, self.hw_aspect_thr)
            if data['path'].endswith('.mp4'):
                is_pick = is_pick and filter_duration_presto(data['num_frames'], self.num_prompts, self.stride_per_prompt)
            pick_list.append(is_pick)
            if i % 1000000 == 0:
                lprint(f'Filter {i}')
        self.data_list = self.data_list.loc[pick_list]
        self.data_list = self.data_list.reset_index(drop=True)
        lprint(f'Data length: {len(self.data_list)}')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.data_list, f)
            lprint(f'Save cache {cache_path}')

    def check_cache(self):
        unique_identifiers = OrderedDict()
        unique_identifiers['class'] = type(self).__name__
        unique_identifiers['data_dir'] = self.data_dir
        unique_identifiers['meta_file'] = self.meta_file
        unique_identifiers['num_frames'] = self.num_frames
        unique_identifiers['hw_thr'] = self.hw_thr
        unique_identifiers['hw_aspect_thr'] = self.hw_aspect_thr
        unique_identifiers['max_height'] = self.max_height
        unique_identifiers['max_width'] = self.max_width
        unique_identifiers['num_prompts'] = self.num_prompts
        unique_identifiers['stride_per_prompt'] = self.stride_per_prompt
        unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        unique_description_hash = hashlib.md5(unique_description.encode('utf-8')).hexdigest()
        path_to_cache = os.path.join(self.cache_dir, 'data_cache')
        os.makedirs(path_to_cache, exist_ok=True)
        cache_path = os.path.join(
                path_to_cache, f'{unique_description_hash}-{type(self).__name__}-filter_cache.pkl'
        )
        return cache_path
