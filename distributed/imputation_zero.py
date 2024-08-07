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
from momentfm.utils.forecasting_metrics import mse, mae
import matplotlib.pyplot as plt


# Hyperparameters
seed=13
epochs = 3
lr = 1e-4
batch_size = 192
grad_accum_step = 1
use_amp = True
use_tensorcore = True
autotune = True
use_fused = True # False for quick start/debug mode
zero_stage = 2
data_stride_len = 512
mask_ratio = 0.3
dtype = torch.float32
amp_dtype = torch.bfloat16  # use float16 for V100 and bfloat16 for A100
# amp_dtype = torch.float16  # use float16 for V100 and bfloat16 for A100

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
  },
}

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

print('Initializing Deepspeed')
# Initialize DeepSpeed Engine
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_zero_config,
)
if not use_fused:
    optimizer.fused = use_fused
print('Initializing Deepspeed DONE')

# rank = dist.get_rank()
# rank = dist.get_rank()
# world_size = dist.get_world_size()

# Number of parameters in the encoder
if rank == 0: print(f"Number of parameters: {num_params}")

# # Dataset
class WellLogDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 task_name: str = "imputation", 
                 data_split: str = "train", 
                 few_shot: int = 5, 
                 forecast_horizon: int = 192
                ):
        self.seq_len = 512
        self.root_dir = root_dir
        self.task_name = task_name
        self.data_split = data_split
        self.few_shot = few_shot
        self.forecast_horizon = forecast_horizon
        with open(root_dir + 'dict_tokens.json', 'r') as file:
            self.mapping = json.load(file)
        self._read_data()

    def __len__(self):
        return len(self.files)

    def _get_borders(self):
        train_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot))
        test_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot, len(self.mapping)))
        return train_mapping, test_mapping

    def _read_data(self):
        train_mapping, test_mapping = self._get_borders()

        if self.data_split == "train":
               self.files = [f for f in train_mapping.values()]
        elif self.data_split == "test":
               self.files = [f for f in test_mapping.values()]
        self.length_timeseries = len(self.files)
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        input_mask = np.ones(self.seq_len)
        data_dict = torch.load(file_name)
        if self.task_name == 'imputation':
            return data_dict['input'].T, input_mask
        elif self.task_name == 'forecast':
            return  data_dict['input'].T, data_dict['label'].T[:, :self.forecast_horizon], input_mask
        else:
            pass
root_dir = '/gpt/data3/KURC/users/kovaledx/alphatools/logs_tokenized/data_processed_512_standard_Aramco/'
# train_dataset = WellLogDataset(root_dir, few_shot=1000)
# sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

train_dataset = WellLogDataset(root_dir, task_name='imputation', data_split="train",  few_shot=14400)
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)


test_dataset = WellLogDataset(root_dir, task_name='imputation', data_split="test",  few_shot=14400)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)


# random generate data:
# num_sample = batch_size * 1280
# n_channels = 7
# data = torch.randn(num_sample, n_channels, data_stride_len)
# mask = torch.randn(num_sample, data_stride_len) > mask_ratio
# from torch.utils.data import TensorDataset
# train_dataset = TensorDataset(data, mask)
# sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

# use tensor core
if use_tensorcore:
    torch.set_float32_matmul_precision('high')

# training loop
loss_val = []
for epoch in range(epochs):
    for step, (batch_x, batch_masks) in enumerate(tqdm(train_loader, total=len(train_loader))):
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
        
        if rank == 0:
            tqdm.write(f"loss: {loss.item()}")
        
        # Backward
        model.backward(loss)
        model.step()

    if epoch % 2 == 0:
        trues, preds, masks = [], [], []
      
        with torch.no_grad():
            for batch_x_val, batch_masks_val in tqdm(test_loader, total=len(test_loader)):
                trues.append(batch_x_val.numpy())
                
                batch_x_val = batch_x_val.to(rank).float()
                n_channels_val = batch_x_val.shape[1]
                
                # Reshape to [batch_size * n_channels, 1, window_size]
                batch_x_val = batch_x_val.reshape((-1, 1, 512)) 
                
                batch_masks_val = batch_masks_val.to(rank).long()
                batch_masks_val = batch_masks_val.repeat_interleave(n_channels_val, axis=0)

                # mask_val = mask_static
                mask_val = mask_generator.generate_mask(
                    x=batch_x_val, input_mask=batch_masks_val).to(rank).long()
        
                if use_amp:
                    with autocast(dtype=amp_dtype):
                        output_val = model(batch_x_val, input_mask=batch_masks_val, mask=mask_val) # [batch_size, n_channels, window_size]
                else:
                    output_val = model(batch_x_val, input_mask=batch_masks_val, mask=mask_val) # [batch_size, n_channels, window_size]
                
                reconstruction_val = output_val.reconstruction.detach().cpu().numpy()
                mask_val = mask_val.detach().squeeze().cpu().numpy()
                
                # Reshape back to [batch_size, n_channels, window_size]
                reconstruction_val = reconstruction_val.reshape((-1, n_channels, 512)) 
                mask_val = mask_val.reshape((-1, n_channels, 512))
                
                preds.append(reconstruction_val)
                masks.append(mask_val)
    
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        masks = np.concatenate(masks)
        # print(type(trues))
        loss_val.append(mse(y=trues.astype(np.float32)[masks==0], y_hat=preds.astype(np.float32)[masks==0], reduction='mean'))
        # print(loss_val)
        torch.cuda.synchronize()

if rank == 0:
    torch.save(model.state_dict(), "/gpt/data3/KURC/users/kovaledx/src/distributed/saved/moment_model.pt")
    print(f"Mean Squarred Error (MSE)={mse(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}")
    print(f"Mean Absolute Error (MAE)={mae(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}")

    for col in range(5):
        print(f"Mean Squarred Error (MSE)={mse(y=trues[:,col,:][masks[:,col,:]==0], y_hat=preds[:,col,:][masks[:,col,:]==0], reduction='mean')}")
        print(f"Mean Abs Error (MAE)={mae(y=trues[:,col,:][masks[:,col,:]==0], y_hat=preds[:,col,:][masks[:,col,:]==0], reduction='mean')}")
        n_channels = trues.shape[1]
        idx = np.random.randint(trues.shape[0])
        # channel_idx =3
        colms = ["GR", "RDEP", "DTC", "NPHI", "RHOB"]

        fig, axs = plt.subplots(n_channels*2, 1, figsize=(10, 2*n_channels))

        for channel_idx in range(n_channels):
            axs[channel_idx*2].set_title(f"Patch={idx}, Channel={colms[channel_idx]}")
            axs[channel_idx*2].plot(trues[idx, channel_idx, :].squeeze(), label='Ground Truth', c='darkblue')
            axs[channel_idx*2].plot(preds[idx, channel_idx, :].squeeze(), label='Predictions', c='red')
            axs[channel_idx*2].legend(fontsize=6)
            axs[channel_idx*2+1].imshow(np.tile(masks[np.newaxis, idx, channel_idx], reps=(8, 1)), cmap='winter')

        plt.tight_layout()
        plt.savefig('zero5well_50_30_all.png')
        # plt.show()

