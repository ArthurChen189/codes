import argparse
import os
import math
import time
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed, write_basic_config
from accelerate import Accelerator
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import neptune
from neptune.utils import stringify_unsupported

'''
Training LLM using Huggingface Accelerate + Deepspeed.
'''

def parse_option():
    parser = argparse.ArgumentParser()
    
    # global args
    parser.add_argument('--per_device_train_batch_size', type = int, default = 4,
                        help = 'batch size per gpu device.')
    parser.add_argument('--block_size', type = int, default = 8192,
                        help = 'block size, i.e., the length of training sequences.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pretrained_model_name_or_path', type = str, default = "bigcode/starcoder")
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 5e-5, help = "5e-5 for pre-training, 5e-6 for fine-tuning.")
    parser.add_argument('--warmup_ratio', type = float, default = 0.0, help = "ratio of total training steps used for a linear warmup from 0 to max lr.")
    parser.add_argument('--checkpointing_steps', type = int, default = 300)
    # parser.add_argument('--tensorboard_log_dir', type = str, default = "./train_logs")
    parser.add_argument('--mode', type = str, default = "pt")
    parser.add_argument('--output_ckpt_dir', type = str, default = "./ckpts")
    parser.add_argument('--save_all_states', action = 'store_true', 
        help = "whether to save states of model, optimizer, and lr scheduler for resuming training, otherwise only model states are saved.")
    
    # neptune args
    parser.add_argument('--neptune_project', type=str, default="your-workspace/your-project")
    parser.add_argument('--neptune_api_token', type=str, required=True)
    parser.add_argument('--neptune_name', type=str, default=None)
    parser.add_argument('--neptune_tags', nargs='*', default=None)

    # args for pre-training
    parser.add_argument('--pt_data_dir', type = str, default = "./data/corpus.bin")
    parser.add_argument('--resume_from_checkpoint', type = str, default = None, 
                            help = "resuming pre-training from a checkpoint")
    parser.add_argument('--resume_tag', type = str, default = None)
    
    # args for supervised fine-tuning
    parser.add_argument('--text2sql_data_dir', type = str, default = "./data/sft_train_text2sql.json")
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)
    
    opt = parser.parse_args()

    return opt

def checkpoint_model_optimizer_scheduler(checkpoint_folder, model, last_global_step, lr_scheduler, accelerator):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "last_global_step": last_global_step,
    }

    accelerator.print("==> saving model and optimizer <==")
    model.save_checkpoint(checkpoint_folder, last_global_step, checkpoint_state_dict)

    accelerator.print("==> saving lr scheduler <==")
    accelerator.save(lr_scheduler.state_dict(), os.path.join(checkpoint_folder, str(last_global_step), "scheduler.pt"))

    print(f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={last_global_step}")
    return

def resume_model_and_optimizer(model, load_dir, tag):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag = tag, load_optimizer_states = True)
    
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict

    return last_global_step

def checkpoint_model(accelerator, model, tokenizer, output_ckpt_dir, last_global_step):    
    '''
    Utility fuction for only checkpointing the model dictionary (i.e., only model parameters)
    '''
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    accelerator.print("checkpointing model state dict at {}".format(ckpt_path))
    unwrapped_model = accelerator.unwrap_model(model)
    # TODO: currently, there is a small bug that saves a full checkpoint data for each shard when enable zero1 and 2. 
    # See https://github.com/microsoft/DeepSpeed/issues/3303. solution: waiting upgrade of accelerate and deepspeed
    unwrapped_model.save_pretrained(
        ckpt_path, 
        is_main_process = accelerator.is_main_process, 
        save_function = accelerator.save,
        state_dict = accelerator.get_state_dict(model),
        max_shard_size = "100GB"
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_path)
    
    return

def sanity_check(input, target, tokenizer):
    print("Start Sanity Check -------->")
    for t, m in zip(input[:-1], target[1:]):
        decoded = tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<-------- End Sanity Check")

def train(opt):
    set_seed(opt.seed)

    accelerator = Accelerator()
    print("accelerator.is_main_process:", accelerator.is_main_process)
    print("accelerator.device:", accelerator.device)

    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    accelerator.print(opt)
    accelerator.print("tokens per batch:", total_batch_size * opt.block_size)
    accelerator.print("sequences per batch:", total_batch_size)

    # We first search for the most recent checkpoint directory, if it doesn't exist, we start a new run
    most_recent_ckpt_dir = None
    if os.path.exists(opt.output_ckpt_dir):
        import glob
        checkpoint_dirs = glob.glob(os.path.join(opt.output_ckpt_dir, "ckpt-*"))
        if checkpoint_dirs:
            # Sort by checkpoint number (assuming format ckpt-{number})
            most_recent_ckpt_dir = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            opt.resume_from_checkpoint = opt.output_ckpt_dir
            # Extract the checkpoint number for the resume_tag
            opt.resume_tag = os.path.basename(most_recent_ckpt_dir).split('-')[-1]
            accelerator.print(f"Found checkpoint directory: {most_recent_ckpt_dir}, loading model from checkpoint...")
            opt.pretrained_model_name_or_path = most_recent_ckpt_dir

    accelerator.print("using LLM from:", opt.pretrained_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path, 
                                                 attn_implementation = "flash_attention_2", 
                                                 torch_dtype = torch.bfloat16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    model.config.use_cache = False # disable use_cache to enable gradient checkpointing
    model.gradient_checkpointing_enable()
    if opt.mode == "pt":
        dataset = PretrainDataset(opt.pt_data_dir, opt.block_size)
        if accelerator.is_main_process:
            sanity_check(dataset[0]["input_ids"], dataset[0]["labels"], tokenizer)
    elif opt.mode == "sft":
        dataset = SFTSQLGenerationDataset(opt.text2sql_data_dir, tokenizer, opt.block_size, "train", opt.table_num, opt.column_num, None)
    else:
        raise ValueError("opt.mode should be in [pt, sft].")
    dataloader = DataLoader(dataset, batch_size = opt.per_device_train_batch_size, shuffle = True, drop_last = True)

    num_total_batches = math.ceil(opt.epochs * math.ceil(len(dataset) / total_batch_size)) # number of total batches
    optimizer = AdamW(model.parameters(), lr = opt.lr, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.1)

    num_warm_up_batches = max(int(num_total_batches * opt.warmup_ratio), 1)
    accelerator.print("num_warm_up_batches:", num_warm_up_batches)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer = optimizer, 
        warmup_epochs = num_warm_up_batches * accelerator.num_processes,
        max_epochs = num_total_batches * accelerator.num_processes, 
        warmup_start_lr = 0.0, 
        eta_min = 0.1 * opt.lr
    )

    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(optimizer, model, dataloader, lr_scheduler)
    print(type(optimizer))
    print(type(model))
    print(type(dataloader))
    print(type(lr_scheduler))

    accumulation_loss = 0
    global_completed_steps = 0
    model.train()
    resume_epoch = 0
    resume_batch_idx = 0

    # Resume training if checkpoint exists
    if most_recent_ckpt_dir:
        # Extract checkpoint number for the resume_tag
        global_completed_steps = resume_model_and_optimizer(model, opt.resume_from_checkpoint, opt.resume_tag)
        
        resume_epoch = global_completed_steps * accelerator.gradient_accumulation_steps // len(dataloader)
        resume_batch_idx = global_completed_steps * accelerator.gradient_accumulation_steps % len(dataloader)

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print("resume training from {}".format(os.path.join(opt.resume_from_checkpoint, opt.resume_tag)))

        # resume lr scheduler
        lr_scheduler.load_state_dict(torch.load(os.path.join(opt.resume_from_checkpoint, opt.resume_tag, "scheduler.pt")))
        accelerator.print("lr scheduler state dict:", lr_scheduler.state_dict())

    st = time.time()

    if accelerator.is_main_process:
        # Initialize Neptune run instead of wandb
        # We save the run_id in the neptune_run_id.txt file, so that we can resume the run without starting a new neptune run
        if most_recent_ckpt_dir:
            accelerator.print("resume neptune run, loading run_id from neptune_run_id.txt...")
            if os.path.exists(os.path.join(opt.output_ckpt_dir, 'neptune_run_id.txt')):
                with open(os.path.join(opt.output_ckpt_dir, 'neptune_run_id.txt'), 'r') as f:
                    run_id = f.read()
            else:
                raise ValueError("neptune_run_id.txt not found")
            run = neptune.init_run(
                with_id=run_id,
                project=opt.neptune_project,
                api_token=opt.neptune_api_token,
                name=opt.neptune_name,
                tags=opt.neptune_tags,
                dependencies="infer",
            )
        else:
            # if we start a new run, we need to create a new neptune run
            run = neptune.init_run(
                project=opt.neptune_project,
                api_token=opt.neptune_api_token,
                name=opt.neptune_name,
                tags=opt.neptune_tags,
                dependencies="infer"
        )
            run_id = run["sys/id"].fetch()
            if os.path.exists(os.path.join(opt.output_ckpt_dir)):
                with open(os.path.join(opt.output_ckpt_dir, 'neptune_run_id.txt'), 'w') as f:
                    f.write(run_id)
            else:
                # create the folder and file, write the run_id
                os.makedirs(os.path.join(opt.output_ckpt_dir), exist_ok=True)
                with open(os.path.join(opt.output_ckpt_dir, 'neptune_run_id.txt'), 'w') as f:
                    f.write(run_id)
        
        # Log parameters to Neptune
        run["config"] = stringify_unsupported(vars(opt))
        run["total_batch_size"] = total_batch_size
    
    for epoch in range(opt.epochs):
        if opt.resume_from_checkpoint and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            continue
        accelerator.print("Start training epoch:", epoch+1)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc = "Training epoch {}".format(epoch+1))):

            if opt.resume_from_checkpoint and resume_batch_idx > batch_idx:
                accelerator.print("skip {}-th batch".format(batch_idx))
                continue
            
            # `accelerator.accumulate(model)` aims to set right `sync_gradients` state based on the recorded training steps
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accumulation_loss += loss.detach().float()
                # when deepspeed is enabled, `accelerator.backward(loss)` is doing optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` examples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                loss_value = accumulation_loss / accelerator.gradient_accumulation_steps
                lr_value = lr_scheduler.get_last_lr()[0]
                
                accelerator.print("GPU 0, step {}, loss {}".format(global_completed_steps, loss_value))
                accelerator.print("GPU 0, step {}, lr state dict:".format(global_completed_steps), lr_scheduler.state_dict())
                accelerator.print("time elapsed: {}".format(time.time()-st))
                st = time.time()

                if accelerator.is_main_process:
                    # Replace wandb logging with Neptune logging
                    run["train/loss"].append(loss_value)
                    run["train/learning_rate"].append(lr_value)
                    run["train/step"].append(global_completed_steps)

                accumulation_loss = 0

                # save checkpoints for each checkpointing_steps total batch size
                if global_completed_steps % opt.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)
                    if opt.save_all_states:
                        # Remove the old checkpoint directory and its contents before saving new ones
                        old_ckpt_dir = os.path.join(opt.output_ckpt_dir, str(opt.resume_tag))
                        if os.path.exists(old_ckpt_dir):
                            import shutil
                            shutil.rmtree(old_ckpt_dir)
                        
                        checkpoint_model_optimizer_scheduler(opt.output_ckpt_dir, model, global_completed_steps, lr_scheduler, accelerator)
                        opt.resume_tag = global_completed_steps

        accelerator.print("in the end of an epoch, save a checkpoint")
        accelerator.wait_for_everyone()
        checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)
        if opt.save_all_states:
            checkpoint_model_optimizer_scheduler(opt.output_ckpt_dir, model, global_completed_steps, lr_scheduler, accelerator)

    # Stop Neptune run instead of wandb
    if accelerator.is_main_process:
        run.stop()

if __name__ == "__main__":
    # write_basic_config() # initialize the distributed environment using accelerate
    opt = parse_option()
    train(opt)
