from typing import Tuple, Optional
import os
import re
import random
import json
import time
import atexit
from collections import defaultdict
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


CT_LABEL_COLS = ['CT_2', 'CT_3', 'CT_4', 'CT_5', 'CT_6', 'CT_7', 'CT_8', 'CT_9', 'CT_10', 'CT_11']

PETCT_REGIONS = [
    "脑", "眼球及眼眶", "鼻腔、鼻旁窦及鼻咽", "口腔及口咽", "喉部及下咽",
    "甲状腺及大唾液腺", "颈部淋巴结", "肺及胸膜", "纵隔及肺门", "心脏及心包膜",
    "食管", "腋窝及胸壁软组织", "乳腺", "肝脏", "胆囊及胆道系统", "脾脏",
    "胰腺", "胃", "肠道", "肾脏", "肾上腺", "输尿管及膀胱", "生殖系统",
    "腹膜后间隙", "腹膜、肠系膜及网膜", "盆腔及腹股沟淋巴结", "骨骼系统",
    "其他肌肉及皮下组织",
]

PETCT_REGION_TO_IDX = {r: i for i, r in enumerate(PETCT_REGIONS)}

MISSING_SEG_REGIONS = {12, 13, 25, 26}


def _load_labels_dict(labels_path):
    df = pd.read_excel(labels_path)
    num_regions = len(PETCT_REGIONS)
    labels_dict = {}
    for exam_id, group in df.groupby('检查号'):
        exam_id = str(exam_id).strip()
        pet_labels = np.full(num_regions, -1, dtype=np.int64)
        ct_labels = np.zeros((num_regions, len(CT_LABEL_COLS)), dtype=np.float32)
        for _, row in group.iterrows():
            region = str(row['解剖区域']).strip()
            if region not in PETCT_REGION_TO_IDX:
                continue
            ridx = PETCT_REGION_TO_IDX[region]
            pet_val = row['PET摄取分级']
            if pd.notna(pet_val):
                try:
                    v = int(float(pet_val))
                    if 1 <= v <= 5:
                        pet_labels[ridx] = v - 1
                except (ValueError, OverflowError):
                    pass
            for ci, col in enumerate(CT_LABEL_COLS):
                ct_labels[ridx, ci] = float(row[col])
        labels_dict[exam_id] = (pet_labels, ct_labels)
    return labels_dict


class PETCTDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        reports_path: str,
        labels_path: str,
        img_size: Tuple[int, int, int] = (192, 192, 336),
        tokenizer=None,
        split: str = "train",
        test_size: int = 800,
        mode: str = "petctclip",
        organ_labels_path: str = None,
        seg_root: str = None,
        preload: bool = False,
    ) -> None:
        df = pd.read_excel(reports_path)
        report_dict = {}
        for _, row in df.iterrows():
            key = str(row['检查号']).strip()
            text = str(row['审核所见']).strip()
            if text and text != 'nan':
                report_dict[key] = text

        self.labels_dict = _load_labels_dict(labels_path)
        self.mode = mode
        self.img_size = img_size
        self.tokenizer = tokenizer

        if mode == "petctclip":
            self.seg_root = seg_root
            # self.num_sampled_sents = num_sampled_sents
            self.report_dict = report_dict

            self.sentences_dict = defaultdict(list)
            with open(organ_labels_path, encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    self.sentences_dict[d['report_id']].append(d)
            for k in self.sentences_dict:
                self.sentences_dict[k].sort(key=lambda x: x['sentence_id'])

            all_folders = sorted(os.listdir(image_root))
            valid_keys = [f for f in all_folders
                          if f in self.labels_dict and f in self.sentences_dict]
            print(f"PETCTDataset(petctclip): {len(valid_keys)} matched cases "
                  f"out of {len(all_folders)} folders")
        else:
            all_folders = sorted(os.listdir(image_root))
            valid_keys = [f for f in all_folders if f in self.labels_dict]
            print(f"PETCTDataset: {len(valid_keys)} matched cases "
                  f"out of {len(all_folders)} folders")

        rng = random.Random(42)
        rng.shuffle(valid_keys)
        if split == "train":
            keys = valid_keys[:-test_size]
        else:
            keys = valid_keys[-test_size:]

        self.items = [(os.path.join(image_root, k), report_dict.get(k, ''), k)
                      for k in keys]
        print(f"PETCTDataset split={split}: {len(self.items)} samples")

        self.global_tokens = {}
        if tokenizer is not None:
            for _, text, exam_id in tqdm(self.items, desc="Tokenizing global reports"):
                if text:
                    tok = tokenizer(text, truncation=True, max_length=1024)
                    self.global_tokens[exam_id] = (
                        torch.tensor(tok['input_ids']),
                        torch.tensor(tok['attention_mask']),
                    )

        self.region_sents_dict = {}
        if mode == "petctclip" and tokenizer is not None:
            for _, _, exam_id in tqdm(self.items, desc="Tokenizing region sentences"):
                sentences = self.sentences_dict.get(exam_id, [])
                region_to_sents = defaultdict(list)
                for sent in sentences:
                    valid_organs = [o for o in sent['organ_labels'] if o not in MISSING_SEG_REGIONS]
                    if not valid_organs:
                        continue
                    tok = tokenizer(sent['text'], truncation=True, max_length=128)
                    tok_pair = (torch.tensor(tok['input_ids']), torch.tensor(tok['attention_mask']))
                    for organ_id in valid_organs:
                        region_to_sents[organ_id].append(tok_pair)
                self.region_sents_dict[exam_id] = dict(region_to_sents)
            del self.sentences_dict
            del self.report_dict

        self._preloaded = False
        if preload and mode == "petctclip":
            self._preload_shared_memory()

    def _shm_name(self, tag):
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        return f"petct_{job_id}_{tag}"

    def _preload_shared_memory(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        N = len(self.items)
        H, W, D = self.img_size
        vol = H * W * D
        ct_nbytes = N * vol * 2
        pet_nbytes = ct_nbytes
        seg_nbytes = N * vol * 1

        sync_dir = f"/tmp/_petctclip_preload_{job_id}"

        if local_rank == 0:
            os.makedirs(sync_dir, exist_ok=True)
            for f in os.listdir(sync_dir):
                os.remove(os.path.join(sync_dir, f))
            for tag in ("ct", "pet", "seg"):
                name = self._shm_name(tag)
                try:
                    old = SharedMemory(name=name)
                    old.close()
                    old.unlink()
                except FileNotFoundError:
                    pass
            self._ct_shm = SharedMemory(name=self._shm_name("ct"), create=True, size=ct_nbytes)
            self._pet_shm = SharedMemory(name=self._shm_name("pet"), create=True, size=pet_nbytes)
            self._seg_shm = SharedMemory(name=self._shm_name("seg"), create=True, size=seg_nbytes)
            open(os.path.join(sync_dir, "shm_ready"), "w").close()
        else:
            while not os.path.exists(os.path.join(sync_dir, "shm_ready")):
                time.sleep(0.1)
            self._ct_shm = SharedMemory(name=self._shm_name("ct"))
            self._pet_shm = SharedMemory(name=self._shm_name("pet"))
            self._seg_shm = SharedMemory(name=self._shm_name("seg"))

        ct_buf = np.ndarray((N, H, W, D), dtype=np.float16, buffer=self._ct_shm.buf)
        pet_buf = np.ndarray((N, H, W, D), dtype=np.float16, buffer=self._pet_shm.buf)
        seg_buf = np.ndarray((N, H, W, D), dtype=np.int8, buffer=self._seg_shm.buf)

        my_indices = list(range(local_rank, N, local_world_size))

        def _load_one(i):
            folder, _, exam_id = self.items[i]
            ct_buf[i] = np.load(os.path.join(folder, 'CT.npy'))
            pet_buf[i] = np.load(os.path.join(folder, 'PET.npy'))
            seg_buf[i] = np.load(os.path.join(self.seg_root, exam_id, 'region.npy'))

        num_threads = max(4, min(16, (os.cpu_count() or 8) // local_world_size))
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            it = pool.map(_load_one, my_indices)
            if local_rank == 0:
                it = tqdm(it, total=len(my_indices), desc="Preloading to shared memory")
            for _ in it:
                pass

        open(os.path.join(sync_dir, f"rank_{local_rank}_done"), "w").close()
        for r in range(local_world_size):
            while not os.path.exists(os.path.join(sync_dir, f"rank_{r}_done")):
                time.sleep(0.5)

        if local_rank == 0:
            print(f"Preload done: {N} samples by {local_world_size} ranks, "
                  f"shm total {(ct_nbytes + pet_nbytes + seg_nbytes) / 1e9:.1f} GB", flush=True)

        self._ct_cache = np.ndarray((N, H, W, D), dtype=np.float16, buffer=self._ct_shm.buf)
        self._pet_cache = np.ndarray((N, H, W, D), dtype=np.float16, buffer=self._pet_shm.buf)
        self._seg_cache = np.ndarray((N, H, W, D), dtype=np.int8, buffer=self._seg_shm.buf)
        self._preloaded = True

        def _cleanup():
            for shm_obj in (self._ct_shm, self._pet_shm, self._seg_shm):
                shm_obj.close()
            if local_rank == 0:
                for shm_obj in (self._ct_shm, self._pet_shm, self._seg_shm):
                    shm_obj.unlink()
        atexit.register(_cleanup)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.mode == "petctclip":
            return self._getitem_petctclip(idx)
        return self._getitem_clip(idx)

    def _getitem_petctclip(self, idx: int):
        folder, text, exam_id = self.items[idx]

        if self._preloaded:
            ct = self._ct_cache[idx].astype(np.float32)
            pet = self._pet_cache[idx].astype(np.float32)
            seg = self._seg_cache[idx]
        else:
            ct = np.load(os.path.join(folder, 'CT.npy')).astype(np.float32)
            pet = np.load(os.path.join(folder, 'PET.npy')).astype(np.float32)
            seg = np.load(os.path.join(self.seg_root, exam_id, 'region.npy'))

        result = {
            'pet': torch.from_numpy(pet[np.newaxis]),
            'ct': torch.from_numpy(ct[np.newaxis]),
        }
        result['seg'] = torch.from_numpy(seg)

        if exam_id in self.global_tokens:
            result['global_input_id'], result['global_attention_mask'] = self.global_tokens[exam_id]

        result['region_to_sentences'] = self.region_sents_dict.get(exam_id, {})

        pet_labels, ct_labels = self.labels_dict[exam_id]
        organ_abnormal = np.zeros(28, dtype=np.int64)
        pet_abnormal = (pet_labels != 1) & (pet_labels != 2)
        ct_abnormal = ct_labels.any(axis=1).astype(bool)
        organ_abnormal[pet_abnormal | ct_abnormal] = 1
        result['organ_labels'] = torch.from_numpy(organ_abnormal)

        return result


