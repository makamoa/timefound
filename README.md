# Model Pre-Training

![image](https://github.com/user-attachments/assets/ea23fd7d-a39f-4649-ae57-35ca35573c4d)

This project provides a script for pre-training of foundation time-series model on well-log data. The script includes features for distributed training, DeepSpeed optimization, checkpointing, and automatic resume functionality for interrupted training sessions.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or later
- PyTorch
- DeepSpeed
- Other dependencies (e.g., `momentfm`, `torchvision`, etc.)

You can install the necessary dependencies via:
```bash
pip install -r requirements.txt
```

## Project Structure

    pretrain.py: The main script for pre-training the MOMENT model.
    dataset.py: Defines the WellLogDataset class for loading the well-log data.
    common.py: Contains utility functions, including get_train_args.
    results/: Directory where model checkpoints and configuration files are saved by default.
    config.json: Optional configuration file to specify hyperparameters and settings.

## Usage

The pretrain.py script supports various command-line arguments for configuring the training process. You can pass these parameters directly via command line or through a JSON configuration file.

| Argument                  | Type    | Description                                                             |
|---------------------------|---------|-------------------------------------------------------------------------|
| `--config`                | str     | Path to JSON config file with hyperparameters.                          |
| `--log_dir`               | str     | Directory to save logs.                                                 |
| `--results_dir`           | str     | Directory to save model checkpoints and config file. Default: `results/`. |
| `--resume_checkpoint`     | str     | Path to checkpoint to resume training.                                  |
| `--seed`                  | int     | Seed for random number generation. Default: `13`.                       |
| `--batch_size`            | int     | Batch size for each device. Default: `2`.                               |
| `--total_token_batch_size`| int     | Total token batch size across devices. Default: `65536`.                |
| `--max_lr`                | float   | Maximum learning rate. Default: `6e-4`.                                 |
| `--min_lr`                | float   | Minimum learning rate after warmup. Default: `6e-5`.                    |
| `--grad_clip_value`       | float   | Gradient clipping value. Default: `1.0`.                                |
| `--warmup_steps`          | int     | Number of warmup steps for learning rate scheduler.                     |
| `--max_steps`             | int     | Maximum number of training steps. Default: `1000`.                      |
| `--max_epochs`            | int     | Maximum number of epochs. Default: `1`.                                 |
| `--check_every_step`      | int     | Log every `n` steps. Default: `1`.                                      |
| `--use_amp`               | bool    | Enable mixed-precision (AMP) training. Default: `True`.                 |
| `--data_stride_len`       | int     | Length of the data stride for each batch. Default: `512`.               |
| `--n_channels`            | int     | Number of data channels in the input. Default: `5`.                     |
| `--mask_ratio`            | float   | Ratio of masked patches for training. Default: `0.3`.                   |
| `--root_dir`              | str     | Path to the data root directory.                                        |
| `--rank`                  | int     | Rank for distributed training. Default: `0`.                            |
| `--world_size`            | int     | Total number of devices for distributed training. Default: `torch.cuda.device_count()`. |


## Running the Pre-Training Script

### Run Directly with Command-Line Arguments

```
python pretrain.py --batch_size 4 --max_epochs 10 --root_dir /path/to/data --results_dir /path/to/results
```
### Run with a Config File

Create a configuration file, e.g., config.json:

{
    "batch_size": 4,
    "max_epochs": 10,
    "root_dir": "/path/to/data",
    "results_dir": "/path/to/results"
}

Then run the script with the --config argument:

```
python pretrain.py --config config.json
```

## Resume Training from Checkpoint

If training is interrupted, use the --resume_checkpoint argument to specify the path to the checkpoint file. This will resume training from the last saved state.

```
python pretrain.py --config config.json --resume_checkpoint /path/to/results/checkpoint_epoch_10.pt
```

## Checkpoints and Logging

The script saves a checkpoint every 10 epochs in the specified results_dir. Checkpoints contain the model and optimizer states, allowing you to resume training from the last checkpoint.
The configuration file is also saved in the results directory, making it easy to restart training with the same settings.

## Additional Notes

This script uses DeepSpeed for distributed training, which is recommended for scaling across multiple GPUs. Ensure that DeepSpeed is correctly installed and configured on your machine.
Use mixed-precision (AMP) training to reduce memory usage and potentially improve performance on supported hardware.
Make sure torch.distributed.launch or a similar distributed launch utility is used when training on multiple nodes.
