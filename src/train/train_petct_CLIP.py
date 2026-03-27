import os
import time
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import transformers
from transformers import Trainer, TrainerCallback, AutoTokenizer
from dataclasses import dataclass, field
import random
import pandas as pd
from safetensors.torch import load_file

from src.dataloader.dataloader import PETCTDataset
from src.model.CLIP import CLIPConfig, CLIP, PETCTCLIPConfig, PETCTCLIP
import swanlab


@dataclass
class ModelArguments:
    root_dir: str = field(default="/data/home/run")
    language_model_name_or_path: str = field(default="models/BioClinical-ModernBERT/BioClinical-ModernBERT-base/")

    model_type: str = field(default="petctclip")

    gather_loss: bool = field(default=True)
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    channels: int = field(default=2)
    img_size: List[int] = field(default_factory=lambda: [192, 192, 336])
    patch_size: List[int] = field(default_factory=lambda: [16, 16, 12])

    dim: int = field(default=768)
    mlp_dim: int = field(default=3072)
    depth: int = field(default=12)
    heads: int = field(default=12)
    dim_head: int = field(default=64)
    dropout: float = field(default=0.0)
    emb_dropout: float = field(default=0.0)

    margin: float = field(default=0.1)
    lambda_loc: float = field(default=1.0)
    lambda_ex: float = field(default=0.5)
    lambda_and: float = field(default=0.3)
    lambda_ent: float = field(default=0.01)
    lambda_pr: float = field(default=0.1)
    pi_0: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.50])


@dataclass
class DataArguments:
    data_root: Optional[str] = field(default=None)
    image_root: str = field(default="data/PETCT/images")
    reports_path: str = field(default="data/PETCT/reports.xlsx")
    labels_path: str = field(default="data/PETCT/aggregated_labels.xlsx")
    organ_labels_path: str = field(default="data/PETCTorgan_labels.jsonl")
    seg_root: str = field(default="data/PETCT/seg")
    abnormality_rates_path: str = field(default="data/PETCT/region_abnormality_rates.xlsx")
    test_size: int = field(default=800)
    num_sampled_regions: int = field(default=4)
    preload: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    save_safetensors: bool = field(default=False)

    bf16: bool = True
    output_dir: str = "projs/PETCTCLIP/output"
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "no"
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.02
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    dataloader_persistent_workers: bool = True
    report_to: str = "swanlab"
    run_name: str = "petct-clip-1"

    stage2_step: int = field(default=5000)
    stage1_lambda_loc: float = field(default=0.1)

    eval_retrieval_steps: int = field(default=500)
    eval_retrieval_samples: int = field(default=512)
    eval_retrieval_batch_size: int = field(default=4)


class EvalCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list) -> dict:
        pet = torch.stack([b['pet'] for b in batch])
        ct = torch.stack([b['ct'] for b in batch])
        ids = [b['global_input_id'] for b in batch]
        masks = [b['global_attention_mask'] for b in batch]
        max_len = max(len(x) for x in ids)
        input_ids = torch.stack([
            F.pad(x, (0, max_len - len(x)), value=self.pad_token_id) for x in ids
        ])
        attn_mask = torch.stack([
            F.pad(x, (0, max_len - len(x)), value=0) for x in masks
        ])
        return dict(pet=pet, ct=ct,
                    global_input_ids=input_ids, global_attention_mask=attn_mask)


class RetrievalEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, collator, eval_steps, num_samples, batch_size):
        self.eval_steps = eval_steps
        rng = random.Random(42)
        indices = list(range(len(eval_dataset)))
        rng.shuffle(indices)
        indices = indices[:num_samples]
        subset = Subset(eval_dataset, indices)
        self.loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            collate_fn=collator, num_workers=4, pin_memory=True,
        )
        self.num_samples = len(indices)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return
        if model is None:
            return

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_main:
            return

        m = model.module if hasattr(model, "module") else model
        m.eval()

        all_v_fus, all_t_global = [], []
        device = next(m.parameters()).device

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for batch in self.loader:
                pet = batch['pet'].to(device)
                ct = batch['ct'].to(device)
                ids = batch['global_input_ids'].to(device)
                mask = batch['global_attention_mask'].to(device)

                F_pet, F_ct = m.encode_vision(pet, ct)
                _, _, v_fus = m.get_global_vision(F_pet, F_ct)
                t_global = m.encode_text_global(ids, mask)

                all_v_fus.append(v_fus.float().cpu())
                all_t_global.append(t_global.float().cpu())

        all_v_fus = torch.cat(all_v_fus, dim=0)
        all_t_global = torch.cat(all_t_global, dim=0)
        N = all_v_fus.shape[0]

        sim = all_t_global @ all_v_fus.T
        gt = torch.arange(N).unsqueeze(1)

        _, topk_t2i = sim.topk(50, dim=1)
        r50_t2i = (topk_t2i == gt).any(dim=1).float().mean().item()

        _, topk_i2t = sim.T.topk(50, dim=1)
        r50_i2t = (topk_i2t == gt).any(dim=1).float().mean().item()

        print(f"[Eval step {state.global_step}] R@50 t2i={r50_t2i:.4f}  i2t={r50_i2t:.4f}")
        try:
            swanlab.log({
                "eval/R@50_t2i": r50_t2i,
                "eval/R@50_i2t": r50_i2t,
            }, step=state.global_step)
        except Exception:
            pass

        m.train()


class FreezeTextCallback(TrainerCallback):
    def __init__(self, freeze_steps=1000):
        self.freeze_steps = freeze_steps
        self.unfrozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        m = model.module if hasattr(model, "module") else model
        for p in m.language_encoder.parameters():
            p.requires_grad = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.unfrozen or model is None:
            return
        if state.global_step >= self.freeze_steps:
            m = model.module if hasattr(model, "module") else model
            for p in m.language_encoder.parameters():
                p.requires_grad = True
            self.unfrozen = True


class TimedDataLoader:
    def __init__(self, dataloader, trainer):
        self.dataloader = dataloader
        self.trainer = trainer

    def __iter__(self):
        it = iter(self.dataloader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            dt = time.perf_counter() - t0
            self.trainer._dl_total_time += dt
            self.trainer._dl_max_time = max(self.trainer._dl_max_time, dt)
            self.trainer._dl_count += 1
            yield batch

    def __len__(self):
        return len(self.dataloader)


class LossLoggingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_logs = {}
        self._custom_log_count = 0
        self._dl_total_time = 0.0
        self._dl_max_time = 0.0
        self._dl_count = 0

    def get_train_dataloader(self):
        return TimedDataLoader(super().get_train_dataloader(), self)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        if "losses" in outputs:
            for k, v in outputs["losses"].items():
                if k == "total":
                    continue
                self._custom_logs[k] = self._custom_logs.get(k, 0.0) + v.detach().float().item()
            self._custom_log_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._custom_log_count > 0:
            for k, v in self._custom_logs.items():
                logs[k] = round(v / self._custom_log_count, 6)
            self._custom_logs = {}
            self._custom_log_count = 0
        if self._dl_count > 0:
            logs['dl_avg'] = round(self._dl_total_time / self._dl_count, 3)
            logs['dl_max'] = round(self._dl_max_time, 3)
            self._dl_total_time = 0.0
            self._dl_max_time = 0.0
            self._dl_count = 0
        super().log(logs, start_time)


class StageCallback(TrainerCallback):
    def __init__(self, stage2_step, stage1_lambda_loc, full_lambda_loc):
        self.stage2_step = stage2_step
        self.stage1_lambda_loc = stage1_lambda_loc
        self.full_lambda_loc = full_lambda_loc
        self.switched = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        m = model.module if hasattr(model, "module") else model
        m.config.lambda_loc = self.stage1_lambda_loc

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.switched or model is None:
            return
        if state.global_step >= self.stage2_step:
            m = model.module if hasattr(model, "module") else model
            m.config.lambda_loc = self.full_lambda_loc
            self.switched = True


@dataclass
class CLIPDataCollator:
    def __init__(self, gather_all, pad_token_id=0):
        self.gather_all = gather_all
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list) -> dict:
        images = torch.stack([b['image'] for b in batch], dim=0)

        input_ids_list = [b['input_id'] for b in batch]
        attention_mask_list = [b['attention_mask'] for b in batch]

        max_len = max(len(ids) for ids in input_ids_list)

        input_ids = torch.stack([
            torch.nn.functional.pad(ids, (0, max_len - len(ids)), value=self.pad_token_id)
            for ids in input_ids_list
        ])
        attention_mask = torch.stack([
            torch.nn.functional.pad(mask, (0, max_len - len(mask)), value=0)
            for mask in attention_mask_list
        ])

        batch_size = images.shape[0]
        if self.gather_all and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


@dataclass
class PETCTCLIPDataCollator:
    def __init__(self, pad_token_id=0, abnormality_rates=None, num_sampled_regions=6):
        self.pad_token_id = pad_token_id
        self.abnormality_rates = abnormality_rates or {}
        self.num_sampled_regions = num_sampled_regions

    def _pad_and_stack(self, id_lists, pad_value):
        max_len = max(len(ids) for ids in id_lists)
        return torch.stack([
            torch.nn.functional.pad(ids, (0, max_len - len(ids)), value=pad_value)
            for ids in id_lists
        ])

    def _sample_regions(self, batch):
        B = len(batch)
        common_regions = set(batch[0]['region_to_sentences'].keys())
        for b in batch[1:]:
            common_regions &= set(b['region_to_sentences'].keys())
        if not common_regions:
            return []

        organ_labels_list = [b['organ_labels'] for b in batch]

        all_normal = []
        has_abnormal = []
        for r in common_regions:
            if all(ol[r - 1].item() == 0 for ol in organ_labels_list):
                all_normal.append(r)
            else:
                has_abnormal.append(r)

        K = min(self.num_sampled_regions, len(common_regions))
        selected = []

        # 先保证至少选一个 has_abnormal 的器官
        if has_abnormal:
            weights = [self.abnormality_rates.get(r, 0.01) for r in has_abnormal]
            pick = random.choices(has_abnormal, weights=weights, k=1)[0]
            selected.append(pick)

        # 剩余名额从所有未选器官中随机采样
        remaining = [r for r in common_regions if r not in selected]
        if remaining and len(selected) < K:
            needed = K - len(selected)
            selected += random.sample(remaining, min(needed, len(remaining)))

        return selected

    def __call__(self, batch: list) -> dict:
        B = len(batch)
        pet = torch.stack([b['pet'] for b in batch], dim=0)
        ct = torch.stack([b['ct'] for b in batch], dim=0)

        global_input_ids = self._pad_and_stack(
            [b['global_input_id'] for b in batch], self.pad_token_id
        )
        global_attention_mask = self._pad_and_stack(
            [b['global_attention_mask'] for b in batch], 0
        )

        selected_regions = self._sample_regions(batch)

        all_sent_ids, all_sent_masks = [], []
        sent_patient_idx, sent_organ_idx = [], []
        all_organ_masks = []

        for region_id in selected_regions:
            for n, b in enumerate(batch):
                candidates = b['region_to_sentences'][region_id]
                input_ids, attn_mask = random.choice(candidates)
                all_sent_ids.append(input_ids)
                all_sent_masks.append(attn_mask)
                sent_patient_idx.append(n)
                sent_organ_idx.append(region_id - 1)
                organ_mask = (b['seg'] == region_id).bool()
                all_organ_masks.append(organ_mask)

        result = dict(
            pet=pet,
            ct=ct,
            global_input_ids=global_input_ids,
            global_attention_mask=global_attention_mask,
        )

        if all_sent_ids:
            result['sent_input_ids'] = self._pad_and_stack(all_sent_ids, self.pad_token_id)
            result['sent_attention_mask'] = self._pad_and_stack(all_sent_masks, 0)
            result['sent_patient_idx'] = torch.tensor(sent_patient_idx, dtype=torch.long)
            result['sent_organ_idx'] = torch.tensor(sent_organ_idx, dtype=torch.long)
            result['organ_masks'] = torch.stack(all_organ_masks, dim=0)

        result['organ_labels'] = torch.stack([b['organ_labels'] for b in batch], dim=0)

        return result


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    root_dir = model_args.root_dir
    data_root = data_args.data_root or root_dir
    model_args.language_model_name_or_path = os.path.join(root_dir, model_args.language_model_name_or_path)
    image_root = os.path.join(data_root, data_args.image_root)
    reports_path = os.path.join(data_root, data_args.reports_path)
    output_dir = os.path.join(root_dir, training_args.output_dir, training_args.run_name)
    training_args.output_dir = output_dir

    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path, trust_remote_code=True)
    labels_path = os.path.join(data_root, data_args.labels_path)

    if model_args.model_type == "petctclip":
        config = PETCTCLIPConfig.from_dict(vars(model_args))
        model = PETCTCLIP(config)

        if model_args.pretrained_model:
            ckpt = load_file(model_args.pretrained_model)
            model.load_state_dict(ckpt, strict=False)
            print("load pretrained model.")

        organ_labels_path = os.path.join(data_root, data_args.organ_labels_path)
        seg_root = os.path.join(data_root, data_args.seg_root)

        rates_csv = pd.read_excel(os.path.join(root_dir, data_args.abnormality_rates_path))
        abnormality_rates = {}
        for idx, row in rates_csv.iterrows():
            organ_id = int(idx)
            pet_r, ct_r = float(row['PET异常率']), float(row['CT异常率'])
            abnormality_rates[organ_id] = 1.0 - (1.0 - pet_r) * (1.0 - ct_r)

        train_dataset = PETCTDataset(
            image_root=image_root,
            reports_path=reports_path,
            labels_path=labels_path,
            img_size=model_args.img_size,
            tokenizer=tokenizer,
            split="train",
            test_size=data_args.test_size,
            mode="petctclip",
            organ_labels_path=organ_labels_path,
            seg_root=seg_root,
            preload=data_args.preload,
        )

        data_collator = PETCTCLIPDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            abnormality_rates=abnormality_rates,
            num_sampled_regions=data_args.num_sampled_regions,
        )
        stage_callback = StageCallback(
            stage2_step=training_args.stage2_step,
            stage1_lambda_loc=training_args.stage1_lambda_loc,
            full_lambda_loc=model_args.lambda_loc,
        )
        freeze_callback = FreezeTextCallback(freeze_steps=training_args.stage2_step)

        eval_dataset = PETCTDataset(
            image_root=image_root,
            reports_path=reports_path,
            labels_path=labels_path,
            img_size=model_args.img_size,
            tokenizer=tokenizer,
            split="test",
            test_size=data_args.test_size,
            mode="petctclip",
            organ_labels_path=organ_labels_path,
            seg_root=seg_root,
            preload=False,
        )
        eval_collator = EvalCollator(pad_token_id=tokenizer.pad_token_id)
        retrieval_eval_callback = RetrievalEvalCallback(
            eval_dataset=eval_dataset,
            collator=eval_collator,
            eval_steps=training_args.eval_retrieval_steps,
            num_samples=training_args.eval_retrieval_samples,
            batch_size=training_args.eval_retrieval_batch_size,
        )

        trainer = LossLoggingTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            callbacks=[freeze_callback, stage_callback, retrieval_eval_callback],
        )

    else:
        config = CLIPConfig.from_dict(vars(model_args))
        model = CLIP(config)

        if model_args.pretrained_model:
            ckpt = load_file(model_args.pretrained_model)
            model.load_state_dict(ckpt, strict=False)
            print("load pretrained model.")

        train_dataset = PETCTDataset(
            image_root=image_root,
            reports_path=reports_path,
            labels_path=labels_path,
            img_size=model_args.img_size,
            tokenizer=tokenizer,
            split="train",
            test_size=data_args.test_size,
        )

        gather_all = model_args.gather_loss and not model_args.local_loss
        data_collator = CLIPDataCollator(gather_all, pad_token_id=tokenizer.pad_token_id)
        freeze_callback = FreezeTextCallback(freeze_steps=2000)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            callbacks=[freeze_callback],
        )

    trainer.train()

    trainer.save_state()
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
