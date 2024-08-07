from pprint import pprint
import os
import json
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
from dataset import WellLogDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics


# Hyperparameters
seed=13
epochs = 3
lr = 1e-4
batch_size = 16
grad_accum_step = 1
use_amp = True
use_tensorcore = True
autotune = False
use_fused = True
zero_stage = 2
data_stride_len = 512
mask_ratio = 0.3
dtype = torch.float32
amp_dtype = torch.bfloat16  # use float16 for V100 and bfloat16 for A100
#amp_dtype = torch.float16  # use float16 for V100 and bfloat16 for A100
forecast_horizon = 192

ds_zero_config = {
  "train_batch_size": batch_size,
  "gradient_accumulation_steps": grad_accum_step,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": lr
    }
  },
  "fp16": {
    "enabled": amp_dtype == torch.float16 and use_amp
  },
  "bf16": {
   "enabled": amp_dtype == torch.bfloat16 and use_amp
  },
  "zero_optimization": {
    "stage": zero_stage
  }
}

control_randomness(seed=seed) # Set random seeds for PyTorch, Numpy etc.

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': forecast_horizon,
        'head_dropout': 0.1,
        'weight_decay': 0,
        'freeze_encoder': False, # Freeze the patch embedding layer
        'freeze_embedder': False, # Freeze the transformer encoder
        'freeze_head': False, # The linear forecasting head must be trained
    },
)

mask_generator = Masking(mask_ratio=mask_ratio) # Mask 30% of patches randomly 

num_params = sum(p.numel() for p in model.parameters())

rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", -1))

model = model.to(rank, dtype)

# Optimize Mean Squarred Error using your favourite optimizer
criterion = torch.nn.MSELoss() 
if use_fused:
    print("Torch compile needs some times...")
    model = torch.compile(model)
    optimizer = FusedAdam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize DeepSpeed Engine
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_zero_config,
)
if not use_fused:
    optimizer.fused = use_fused

# rank = dist.get_rank()
# rank = dist.get_rank()
# world_size = dist.get_world_size()

# Number of parameters in the encoder
if rank == 0: print(f"Number of parameters: {num_params}")

root_dir = '/gpt/data3/KURC/users/kovaledx/alphatools/logs_tokenized/data_processed_512_standard_Aramco/'
train_dataset = WellLogDataset(root_dir, task_name='forecast', data_split="train", forecast_horizon=forecast_horizon, few_shot=14000)
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

test_dataset = WellLogDataset(root_dir, task_name='forecast', data_split="test", forecast_horizon=forecast_horizon, few_shot=14000)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

# use tensor core
if use_tensorcore:
    torch.set_float32_matmul_precision('high')

# training loop
for epoch in range(epochs):
    losses = []

    model.train()
    for timeseries, forecast, input_mask in tqdm(train_dataloader, total=len(train_dataloader)):
        # Move the data to the GPU
        timeseries = timeseries.float().to(rank)
        input_mask = input_mask.to(rank)
        forecast = forecast.float().to(rank)

        with torch.cuda.amp.autocast():
            output = model(timeseries, input_mask)
            print(output)

        
        loss = criterion(output.forecast, forecast)

        # Scales the loss for mixed precision training
        # Backward
        model.backward(loss)
        model.step()
        torch.cuda.synchronize()
        
        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")
        
        
    if epoch % 2 == 0 :
        model.eval()
        with torch.no_grad():
            for timeseries, forecast, input_mask in tqdm(test_dataloader, total=len(test_dataloader)):
            # Move the data to the GPU
                timeseries = timeseries.float().to(rank)
                input_mask = input_mask.to(rank)
                forecast = forecast.float().to(rank)

                with torch.cuda.amp.autocast():
                    output = model(timeseries, input_mask)
                
                loss = criterion(output.forecast, forecast)                
                losses.append(loss.item())

                trues.append(forecast.detach().cpu().numpy())
                preds.append(output.forecast.detach().cpu().numpy())
                histories.append(timeseries.detach().cpu().numpy())
        
        losses = np.array(losses)
        average_loss = np.average(losses)
        model.train()

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)
        
        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
