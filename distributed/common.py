def get_train_args(num_samples, tokens_per_sample, global_token_batch_size, 
                   batch_size_per_dprank_per_micro_step, num_dprank, 
                   max_steps=-1, max_epochs=-1):
    if max_steps == -1 and max_epochs == -1:
        raise ValueError("At least one of max_steps or max_epochs must be provided.")
    
    if global_token_batch_size % (batch_size_per_dprank_per_micro_step * tokens_per_sample * num_dprank) != 0:
        raise ValueError("global_token_batch_size must be divisible by "
                         "batch_size_per_dprank_per_micro_step * tokens_per_sample * num_dprank.")

    tokens_per_dprank_per_step = tokens_per_sample * batch_size_per_dprank_per_micro_step
    total_tokens_per_micro_step = tokens_per_dprank_per_step * num_dprank
    grad_accum_steps = global_token_batch_size // total_tokens_per_micro_step
    total_tokens_per_step = total_tokens_per_micro_step * grad_accum_steps
    total_tokens_in_dataset = num_samples * tokens_per_sample

    print('DEBUG: ', tokens_per_dprank_per_step, total_tokens_per_micro_step, grad_accum_steps, total_tokens_per_step, total_tokens_in_dataset)


    if max_steps != -1 and max_epochs != -1:
        calculated_max_steps = (max_epochs * total_tokens_in_dataset) // global_token_batch_size
        calculated_max_epochs = (max_steps * global_token_batch_size) / total_tokens_in_dataset

        if not (calculated_max_steps == max_steps and
                int(calculated_max_epochs) == max_epochs):
            raise ValueError(f"Inconsistent max_steps and max_epochs based on the dataset and "
                             f"configuration. Calculated max_steps from max_epochs: {calculated_max_steps}, "
                             f"provided max_steps: {max_steps}. "
                             f"Calculated max_epochs from max_steps: {int(calculated_max_epochs)}, "
                             f"provided max_epochs: {max_epochs}.")
    elif max_steps == -1:
        max_steps = (max_epochs * total_tokens_in_dataset) // global_token_batch_size
    elif max_epochs == -1:
        max_epochs = (max_steps * global_token_batch_size) / total_tokens_in_dataset

    result = {
        "epochs": max_epochs,
        "max_steps": max_steps,
        "grad_accum_steps": grad_accum_steps,
        "total_tokens_per_step": total_tokens_per_step,
        "total_tokens_per_micro_step": total_tokens_per_micro_step
    }

    return result