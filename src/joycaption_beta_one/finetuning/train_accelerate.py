#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import random
import json
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TVF
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb
import omegaconf
import click
import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, LlavaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from .utils import get_cosine_schedule_with_warmup, temprngstate, log_rank_0
from accelerate import Accelerator

DTYPE_MAP = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}

@dataclass
class Config:
    output_dir: Path = Path("checkpoints")
    wandb_project: Optional[str] = None
    device_batch_size: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_samples: int = 0
    max_samples: int = 400000
    num_epochs: Optional[int] = None
    grad_scaler: bool = False
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.0
    allow_tf32: bool = True
    seed: int = 42
    num_workers: int = 2
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.00
    clip_grad_norm: Optional[float] = 1.0
    dataset: str = "dataset.json"
    images_path: Path = Path(".")
    finetune: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    gradient_checkpointing: bool = True
    test_size: int = 128
    grad_scaler_init: float = 2**16
    text_model_dtype: str = "bfloat16"
    pre_test: bool = True
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0

class ImageDataset(Dataset):
    def __init__(self, examples: list, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, image_token_id: int, image_seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        pixel_values = ex['pixel_values']
        messages = ex['messages']
        convo = [{"role": "system", "content": "You are a helpful image captioner."}] + messages
        convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False)
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                input_tokens.extend([self.image_token_id] * self.image_seq_length)
            else:
                input_tokens.append(token)
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        # mask out everything before assistant response
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        end_header = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_indices = (input_ids == eot_id).nonzero(as_tuple=True)[0]
        header_indices = (input_ids == end_header).nonzero(as_tuple=True)[0]
        assert len(eot_indices) == 3 and len(header_indices) == 3
        start = header_indices[-1] + 1
        labels[:start] = -100
        return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def collate_fn(self, batch):
        batch = [b for b in batch if b['pixel_values'] is not None]
        max_len = max(b['input_ids'].size(0) for b in batch)
        pad = lambda x, n, v: torch.nn.functional.pad(x, (0, n), value=v)
        input_ids = torch.stack([pad(b['input_ids'], max_len - b['input_ids'].size(0), self.pad_token_id) for b in batch])
        attention_mask = torch.stack([pad(b['attention_mask'], max_len - b['attention_mask'].size(0), 0) for b in batch])
        labels = torch.stack([pad(b['labels'], max_len - b['labels'].size(0), -100) for b in batch])
        pixel_values = torch.stack([b['pixel_values'] for b in batch])
        return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def build_datasets_and_loaders(config, tokenizer, model_config, accelerator):
    # load and preprocess data
    dataset_path = Path(config.dataset)
    is_parquet = False
    if dataset_path.is_file():
        with open(dataset_path, 'rb') as f:
            # Check for parquet magic bytes
            if f.read(4) == b'PAR1':
                is_parquet = True

    if is_parquet:
        tqdm.write("Loading preprocessed data from parquet file...")
        table = pq.read_table(config.dataset)
        data = table.to_pylist()
        for ex in tqdm(data, desc="Deserializing tensors"):
            buffer = io.BytesIO(ex['pixel_values'])
            ex['pixel_values'] = torch.load(buffer)
    else:
        data = json.loads(Path(config.dataset).read_text())
        for ex in data:
            ex['messages'] = [{**m, 'content': m['content'].replace('<image>', '').strip()} for m in ex['messages']]

        for ex in tqdm(data, desc="Preprocessing images"):
            image_filename = ex['images'][0]
            img = Image.open(config.images_path / image_filename).convert('RGB')
            if img.size != (384, 384): img = img.resize((384, 384), Image.LANCZOS)
            pixel_values = TVF.pil_to_tensor(img)
            ex['pixel_values'] = pixel_values

    random.shuffle(data)
    test_ex, train_ex = data[:config.test_size], data[config.test_size:]
    train_ds = ImageDataset(train_ex, tokenizer, model_config.image_token_index, model_config.image_seq_length)
    test_ds = ImageDataset(test_ex, tokenizer, model_config.image_token_index, model_config.image_seq_length)
    train_loader = DataLoader(train_ds, batch_size=config.device_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, collate_fn=train_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.device_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=test_ds.collate_fn)
    return train_loader, test_loader, len(train_ds)

class Trainer:
    def __init__(self, config, accelerator, logger):
        self.config, self.accelerator, self.logger = config, accelerator, logger
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # build model
        tokenizer = AutoTokenizer.from_pretrained(config.finetune, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(config.finetune, torch_dtype=DTYPE_MAP[config.text_model_dtype])
        if config.gradient_checkpointing: model.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
        model = get_peft_model(model, lora_cfg)
        self.tokenizer, self.model = tokenizer, model
        # optimizer
        optim_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(optim_params, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps, weight_decay=config.adam_weight_decay)
        # datasets
        train_loader, test_loader, train_set_size = build_datasets_and_loaders(config, tokenizer, model.config, accelerator)
        # determine num_epochs
        if config.num_epochs:
            self.num_epochs = config.num_epochs
        else:
            self.num_epochs = math.ceil(config.max_samples / train_set_size)
        # prepare model, optimizer, dataloaders
        self.model, self.optimizer, self.train_loader, self.test_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, test_loader
        )
        # scheduler steps, now using prepared dataloader
        grad_accum_steps = config.batch_size // (config.device_batch_size * accelerator.num_processes)
        steps_per_epoch = len(self.train_loader) // grad_accum_steps
        total_steps = self.num_epochs * steps_per_epoch
        self.total_steps = total_steps
        num_warmup = math.ceil(config.warmup_samples / config.batch_size)
        if config.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps, min_lr_ratio=config.min_lr_ratio)
        else:
            lr_scheduler = get_scheduler(config.lr_scheduler_type, self.optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps)
        self.lr_scheduler = accelerator.prepare(lr_scheduler)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.grad_scaler, init_scale=config.grad_scaler_init)
        self.global_steps = 0
        self.global_samples = 0

    def run_model(self, batch):
        pixel = batch['pixel_values'].to(self.accelerator.device) / 255.0
        pixel = TVF.normalize(pixel, [0.5], [0.5]).to(torch.bfloat16)
        input_ids = batch['input_ids'].to(self.accelerator.device)
        mask = batch['attention_mask'].to(self.accelerator.device)
        labels = batch['labels'].to(self.accelerator.device)
        outputs = self.model(input_ids=input_ids[:, :-1], pixel_values=pixel, attention_mask=mask[:, :-1], use_cache=False)
        logits = outputs.logits.reshape(-1, outputs.logits.size(-1))
        loss = F.cross_entropy(logits, labels[:, 1:].reshape(-1), reduction='mean')
        return loss

    def train(self):
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.logger.info("Starting training...")
        self.logger.info(f"  Output dir: {self.config.output_dir}")
        self.logger.info(f"  Num epochs: {self.num_epochs}")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Device batch size: {self.config.device_batch_size}")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        if wandb.run:
            wandb.watch(self.model)

        progress_bar = tqdm(range(self.total_steps), desc="training")
        for epoch in range(self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            for step, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    loss = self.run_model(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if self.config.clip_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_steps += 1
                        self.global_samples += self.config.device_batch_size * self.accelerator.num_processes
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            self.logger.info(f"Epoch {epoch+1} finished. Saving checkpoint.")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = self.config.output_dir / f"epoch_{epoch+1}"
            unwrapped_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.validate()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(self.test_loader, desc="eval"):
            loss = self.run_model(batch)
            total_loss += loss.item()
        avg = total_loss / len(self.test_loader)
        self.logger.info(f"Validation loss: {avg}")
        if wandb.run:
            wandb.log({"eval/loss": avg, "global_steps": self.global_steps})
        self.model.train()

def run_training(config: Config):
    """Fine-tunes a model with Accelerate, callable from notebooks."""
    # Logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if config.wandb_project:
        wandb.init(project=config.wandb_project, config=omegaconf.OmegaConf.to_container(config, resolve=True))
    
    accelerator = Accelerator()
    trainer = Trainer(config, accelerator, logger)
    trainer.train()

@click.command()
@click.option('--output-dir', default="checkpoints", type=click.Path(file_okay=False, path_type=Path), help="Output directory for checkpoints.")
@click.option('--wandb-project', default=None, type=str, help="Wandb project name.")
@click.option('--device-batch-size', default=1, type=int, help="Batch size per device.")
@click.option('--batch-size', default=32, type=int, help="Total batch size.")
@click.option('--learning-rate', default=5e-5, type=float, help="Learning rate.")
@click.option('--warmup-samples', default=0, type=int, help="Number of warmup samples.")
@click.option('--max-samples', default=400000, type=int, help="Maximum number of samples to train on (used if num_epochs is not set).")
@click.option('--num-epochs', default=None, type=int, help="Maximum number of epochs to train for. Overrides max_samples.")
@click.option('--grad-scaler', is_flag=True, default=False, help="Use gradient scaler.")
@click.option('--lr-scheduler-type', default="cosine", type=str, help="Learning rate scheduler type.")
@click.option('--min-lr-ratio', default=0.0, type=float, help="Minimum learning rate ratio for cosine scheduler.")
@click.option('--allow-tf32/--no-allow-tf32', default=True, help="Allow TF32 for matmuls.")
@click.option('--seed', default=42, type=int, help="Random seed.")
@click.option('--num-workers', default=2, type=int, help="Number of dataloader workers.")
@click.option('--optimizer-type', default="adamw", type=str, help="Optimizer type.")
@click.option('--adam-beta1', default=0.9, type=float, help="AdamW beta1.")
@click.option('--adam-beta2', default=0.999, type=float, help="AdamW beta2.")
@click.option('--adam-eps', default=1e-8, type=float, help="AdamW epsilon.")
@click.option('--adam-weight-decay', default=0.0, type=float, help="AdamW weight decay.")
@click.option('--clip-grad-norm', default=1.0, type=float, help="Clip gradient norm. Use 0 to disable.")
@click.option('--dataset', default="dataset.json", type=str, help="Path to the dataset JSON file or a preprocessed .parquet file.")
@click.option('--images-path', default=".", type=click.Path(file_okay=False, path_type=Path), help="Path to the images directory.")
@click.option('--finetune', default="fancyfeast/llama-joycaption-beta-one-hf-llava", type=str, help="Model to finetune.")
@click.option('--gradient-checkpointing/--no-gradient-checkpointing', default=True, help="Enable gradient checkpointing.")
@click.option('--test-size', default=128, type=int, help="Number of samples for the test set.")
@click.option('--grad-scaler-init', default=2**16, type=float, help="Initial scale for gradient scaler.")
@click.option('--text-model-dtype', default="bfloat16", type=str, help="Dtype for the text model.")
@click.option('--pre-test/--no-pre-test', default=True, help="Run a validation loop before training. (Currently no-op)")
@click.option('--lora-r', default=64, type=int, help="LoRA r.")
@click.option('--lora-alpha', default=64, type=int, help="LoRA alpha.")
@click.option('--lora-dropout', default=0.0, type=float, help="LoRA dropout.")
def main(**kwargs):
    """Fine-tunes a model with Accelerate."""
    config = omegaconf.OmegaConf.structured(Config(**kwargs))
    run_training(config)

if __name__ == "__main__":
    main()
