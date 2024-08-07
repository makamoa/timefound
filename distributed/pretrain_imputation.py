from pprint import pprint
import os
import json
import time
import math
import itertools
import numpy as np
from contextlib import nullcontext
import torch
from torch.cuda.amp import autocast
from momentfm.data.informer_dataset import InformerDataset
from torch.utils.data import Dataset, DataLoader
import momentfm
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from tqdm.auto import tqdm
import deepspeed
from deepspeed.ops.adam import FusedAdam
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from momentfm.utils.utils import control_randomness
from common import get_train_args
from dataset import WellLogDataset

# Hyperparameters
log_dir = "log"
seed=13
batch_size = 2 # TODO: check
total_token_batch_size = 2**16 * 5 # ~0.5M, in number of tokens
max_lr = 6e-4
min_lr = max_lr * 0.1
weight_decay = 0.1
grad_clip_value = 1.0
warmup_steps = 715
max_steps = -1
max_epochs = 1  # Train 1 epoch
check_every_step = 1
val_every_steps = 250
save_every_steps = 5000
max_batch_size_per_device = 4
use_amp = True
use_tensorcore = True
autotune = True
use_fused = True
zero_stage = 2
data_stride_len = 512
n_channels = 5 
mask_ratio = 0.3
dtype = torch.float32
amp_dtype = torch.bfloat16  # use float16 for V100 and bfloat16 for A100

total_batch_size = total_token_batch_size//n_channels//data_stride_len
ds_zero_config = {
  "train_batch_size": total_batch_size,
  "gradient_accumulation_steps": total_batch_size / max_batch_size_per_device,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": max_lr
    },
    "betas": [
      0.9,
      0.95
    ],
    "weight_decay": 1e-1
  },
  "fp16": {
    "enabled": amp_dtype == torch.float16 and use_amp
  },
  "bf16": {
   "enabled": amp_dtype == torch.bfloat16 and use_amp
  },
  "zero_optimization": {
    "stage": zero_stage
  }, 
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": min_lr,
      "warmup_max_lr": max_lr,
      "warmup_num_steps": warmup_steps
    }
  }
}

# # -----------------------------------------------------------------------------
# Init

control_randomness(seed=seed) # Set random seeds for PyTorch, Numpy etc.

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={'task_name': 'reconstruction', # For imputation, we will load MOMENT in `reconstruction` mode
                   'freeze_encoder': False, # Freeze the patch embedding layer
                   'freeze_embedder': False, # Freeze the transformer encoder
                   'freeze_head': False, # The linear forecasting head must be trained
                 }
)

mask_generator = Masking(mask_ratio=mask_ratio) # Mask 30% of patches randomly 

num_params = sum(p.numel() for p in model.parameters())

rank = int(os.environ.get("LOCAL_RANK", 0))
model = model.to(rank, dtype)

# Optimize Mean Squarred Error using your favourite optimizer
criterion = torch.nn.MSELoss() 
if use_fused:
    print("Torch compile needs some times...")
    model = torch.compile(model)
    optimizer = FusedAdam(model.parameters(), lr=max_lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

# Initialize DeepSpeed Engine
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_zero_config,
)
if not use_fused:
    optimizer.fused = use_fused

# rank = dist.get_rank()
rank = dist.get_rank()
world_size = dist.get_world_size()
master_process = rank == 0

# Number of parameters in the encoder
if master_process: print(f"Number of parameters: {num_params}")

# use tensor core
if use_tensorcore:
    torch.set_float32_matmul_precision('high')

# create the log directory we will write checkpoints to and log to
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

assert total_token_batch_size % (max_batch_size_per_device * n_channels * data_stride_len * world_size) == 0, "make sure total_batch_size is divisible by B * C * T * world_size"

# -----------------------------------------------------------------------------

root_dir = '/gpt/data3/KURC/users/kovaledx/alphatools/logs_tokenized/data_processed_512_standard_Aramco/'
train_dataset = WellLogDataset(root_dir, few_shot=16000)
sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_dataloader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler))

# # -----------------------------------------------------------------------------

training_args = get_train_args(
    len(train_dataset),
    n_channels * data_stride_len,
    total_token_batch_size,
    max_batch_size_per_device,
    world_size,
    max_steps,
    max_epochs,
)

print(training_args)

epochs, max_steps, grad_accum_steps, total_tokens_per_step = \
    training_args['epochs'], training_args['max_steps'], training_args['grad_accum_steps'], training_args['total_tokens_per_step']
if master_process:
    print(f"The training process will train {epochs} epochs, {max_steps} steps.")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"=> calculated tokens per step: {total_tokens_per_step}")

# -----------------------------------------------------------------------------
# simple launch:
# Single device: python pretrain_imputation.py
# DeepSpeed launch for single node:
# deepspeed pretrain_imputation.py

# training loop
for step in tqdm(range(max_steps)):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation Step
    model.eval()
    ...

    # Micro Batch
    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        batch_x, batch_masks = next(train_dataloader)
        batch_x, batch_masks = batch_x.to(rank, dtype), batch_masks.to(rank)
        n_channels = batch_x.shape[1]
        
        # Reshape to [batch_size * n_channels, 1, window_size]
        batch_x = batch_x.reshape((-1, 1, data_stride_len)) 
        
        batch_masks = batch_masks.to(rank).long()
        batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
        
        # Randomly mask some patches of data
        mask = mask_generator.generate_mask(
            x=batch_x, input_mask=batch_masks).to(rank).long()

        if use_amp:
            with autocast(dtype=amp_dtype):
                output = model(batch_x, input_mask=batch_masks, mask=mask)
        else:
            output = model(batch_x, input_mask=batch_masks, mask=mask)
        
        # Compute loss
        recon_loss = criterion(output.reconstruction, batch_x)
        observed_mask = batch_masks * (1 - mask)
        masked_loss = observed_mask * recon_loss
        
        loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
        loss_accum += loss.detach()
        
        # Backward
        model.backward(loss)
    if world_size > 1:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
    else:
        norm = 1
    model.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    
    # log
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    if master_process and (step+1)%check_every_step==0:
        tqdm.write(f"step {step:5d} | loss: {loss_accum.item():.6f} | grad norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {total_tokens_per_step:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    
