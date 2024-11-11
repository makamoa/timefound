import os
import time
import json
import torch
import argparse
import deepspeed
from pprint import pprint
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from common import get_train_args
from dataset import WellLogDataset
from momentfm.utils.utils import control_randomness
from deepspeed.ops.adam import FusedAdam
import torch.distributed as dist

def initialize_hyperparameters(args):
    """Define and parse the hyperparameters."""
    control_randomness(seed=args.seed)
    
    # Set up model, masking, and criterion
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False,
            'freeze_embedder': False,
            'freeze_head': False
        }
    ).to(args.rank, args.dtype)
    
    mask_generator = Masking(mask_ratio=args.mask_ratio)
    criterion = torch.nn.MSELoss()
    
    # Initialize optimizer
    if args.use_fused:
        optimizer = FusedAdam(model.parameters(), lr=args.max_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr)

    return model, mask_generator, criterion, optimizer

def initialize_deepspeed(model, optimizer, args):
    """Initialize DeepSpeed engine with the model and configuration."""
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=args.ds_zero_config,
    )
    return model, optimizer

def setup_data_loader(args):
    """Set up the data loader for training."""
    dataset = WellLogDataset(args.root_dir, few_shot=16000)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    return iter(DataLoader(dataset, batch_size=args.batch_size, sampler=sampler))

def save_checkpoint(model, optimizer, epoch, args):
    """Save model and optimizer states to resume training."""
    checkpoint_path = os.path.join(args.results_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

def load_checkpoint(model, optimizer, args):
    """Load model and optimizer states to resume training."""
    checkpoint_path = args.resume_checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def training_step(model, mask_generator, criterion, optimizer, batch, args):
    """Perform a single training step."""
    batch_x, batch_masks = batch
    batch_x, batch_masks = batch_x.to(args.rank, args.dtype), batch_masks.to(args.rank)
    batch_x = batch_x.reshape((-1, 1, args.data_stride_len))
    batch_masks = batch_masks.repeat_interleave(batch_x.shape[1], axis=0).to(args.rank).long()
    
    mask = mask_generator.generate_mask(x=batch_x, input_mask=batch_masks).to(args.rank).long()

    with autocast(dtype=args.amp_dtype) if args.use_amp else nullcontext():
        output = model(batch_x, input_mask=batch_masks, mask=mask)
    
    recon_loss = criterion(output.reconstruction, batch_x)
    observed_mask = batch_masks * (1 - mask)
    masked_loss = (observed_mask * recon_loss).nansum() / (observed_mask.nansum() + 1e-7)
    
    model.backward(masked_loss)
    optimizer.step()

    return masked_loss

def main(args):
    # Save configuration to results directory
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    model, mask_generator, criterion, optimizer = initialize_hyperparameters(args)
    model, optimizer = initialize_deepspeed(model, optimizer, args)
    data_loader = setup_data_loader(args)

    start_epoch = load_checkpoint(model, optimizer, args)
    for epoch in range(start_epoch, args.max_epochs):
        for step in tqdm(range(args.max_steps)):
            t0 = time.time()
            batch = next(data_loader)
            loss = training_step(model, mask_generator, criterion, optimizer, batch, args)
            t1 = time.time()
            
            if args.master_process and (step + 1) % args.check_every_step == 0:
                print(f"Step {step} | Loss: {loss.item()} | Time: {(t1 - t0) * 1000:.2f} ms")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain MOMENT Model")
    parser.add_argument('--config', type=str, help="Path to config file")
    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--total_token_batch_size', type=int, default=2**16 * 5)
    parser.add_argument('--max_lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip_value', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=715)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--check_every_step', type=int, default=1)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--use_tensorcore', type=bool, default=True)
    parser.add_argument('--use_fused', type=bool, default=True)
    parser.add_argument('--zero_stage', type=int, default=2)
    parser.add_argument('--data_stride_len', type=int, default=512)
    parser.add_argument('--n_channels', type=int, default=5)
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float32)
    parser.add_argument('--amp_dtype', type=torch.dtype, default=torch.bfloat16)
    parser.add_argument('--root_dir', type=str, default='/path/to/data')
    parser.add_argument('--rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                setattr(args, key, value)

    main(args)
