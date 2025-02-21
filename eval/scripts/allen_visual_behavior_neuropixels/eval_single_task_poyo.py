import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import random

import omegaconf
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from kirby.data import Dataset, collate
import kirby.taxonomy
from kirby.taxonomy import (
    decoder_registry,
    Decoder, OutputType, Task
)
from kirby.models import POYOPlusTokenizer
from kirby.transforms import Compose
from kirby.data.sampler import SequentialFixedWindowSampler
from kirby.utils import train_wrapper
from torch.utils.data import DataLoader

from torch_optimizer import Lamb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
import lightning

from sklearn.metrics import accuracy_score, r2_score

from eval.scripts.allen_visual_behavior_neuropixels.eval_utils import bin_target

import logging
logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision("medium")

def move_to_gpu(d, device):
    for k, v in d.items():
        if isinstance(v, dict):
            move_to_gpu(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)

def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:,:,i] = (arr == _uni)
    return ret

# -------
# SET UP
# -------
ap = argparse.ArgumentParser()
ap.add_argument("--session_id", type=str, default="715093703")
ap.add_argument("--behavior", type=str, default="gabors")
ap.add_argument("--model_path", type=str, default="../../../logs/lightning_logs/")
ap.add_argument("--ckpt_name", type=str, default="2xpi3x0u")
ap.add_argument("--save_path", type=str, default="../../../results/")
ap.add_argument("--data_path", type=str, default="/projects/bcxj/yzhang39/allen/datasets/processed/")
ap.add_argument("--config_path", type=str, default="../../../configs/")
ap.add_argument("--config_name", type=str, default="train_allen_gabors.yaml")
args = ap.parse_args()

save_path = args.save_path
data_path = args.data_path
config_path = args.config_path
model_path = args.model_path
config_name = args.config_name
ckpt_name = args.ckpt_name

session_id = args.session_id
behavior = args.behavior

logging.info(f"Evaluating session: {session_id}")

params = {"interval_len": 1., "binsize": 0.02}

# ---------
# LOAD DATA
# ---------

dataset = Dataset(
    root=data_path,
    split="test", # "train"/"valid"/"test"
    include=[{
        "selection": [{
            "dandiset": f"allen_visual_behavior_neuropixels_{behavior}",
            "sortsets": {f"mouse_{session_id}"},
        }],
    }],
)

# ----------
# LOAD MODEL
# ----------

with initialize(version_base="1.3", config_path=config_path, job_name="test_app"):
    cfg = compose(config_name=config_name)

model = hydra.utils.instantiate(
    cfg.model,
    task_specs=decoder_registry,
    backend_config=cfg.backend_config,
    _convert_="object",
)

sequence_length = params["interval_len"]

transforms = hydra.utils.instantiate(
    cfg.train_transforms, sequence_length=sequence_length
)

def list_checkpoints(model_path, ckpt_name):
    ckpt_dir = f"{model_path}/{ckpt_name}"
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and f != "last.ckpt"]
    return checkpoints

checkpoints = list_checkpoints(model_path, ckpt_name)
print("available checkpoints:", checkpoints)

chosen_ckpt = checkpoints[0] if len(checkpoints) > 0 else "last.ckpt"
ckpt = torch.load(f"{model_path}/{ckpt_name}/{chosen_ckpt}", map_location="cpu")
state_dict = ckpt["state_dict"]        

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[6:]
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)

tokenizer = POYOPlusTokenizer(
    model.unit_emb.tokenizer,
    model.session_emb.tokenizer,
    decoder_registry=decoder_registry,
    latent_step=1 / 8,
    num_latents_per_step=cfg.model.num_latents,
    batch_type=model.batch_type,
)

transform = Compose([*transforms, tokenizer])

val_tokenizer = tokenizer
val_tokenizer.eval = True
val_dataset = Dataset(
    cfg.data_root,
    "test",
    include=OmegaConf.to_container(cfg.dataset),  # converts to native list[dicts]
    transform=val_tokenizer,
)
val_sampler = SequentialFixedWindowSampler(
    interval_dict=val_dataset.get_sampling_intervals(),
    window_length=sequence_length,
    step=sequence_length / 2,
)

val_loader = DataLoader(
    val_dataset,
    sampler=val_sampler,
    collate_fn=collate,
    batch_size=cfg.get(
        "eval_batch_size", cfg.batch_size
    ),  # Default to training batch size, but allow override in config.
    num_workers=1,
)

max_lr = cfg.base_lr * cfg.batch_size

epochs = cfg.epochs

optimizer = Lamb(
    model.parameters(),  # filter(lambda p: p.requires_grad, model.parameters()),
    lr=max_lr,
    weight_decay=cfg.weight_decay,
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=epochs,
    steps_per_epoch=len(val_loader),
    pct_start=cfg.pct_start,
    anneal_strategy="cos",
    div_factor=1,
)

wrapper = train_wrapper.TrainWrapper(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)

callbacks = [
    ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
    ModelCheckpoint(
        dirpath=f"../../../logs/lightning_logs/",
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=cfg.eval_epochs,
    ),
    train_wrapper.CustomValidator(val_loader),
    LearningRateMonitor(
        logging_interval="step"
    ),  # Create a callback to log the learning rate.
]

trainer = lightning.Trainer(
    # logger=[tb, wandb],
    default_root_dir=cfg.log_dir,
    check_val_every_n_epoch=cfg.eval_epochs,
    max_epochs=epochs,
    log_every_n_steps=1,
    strategy=(
        "auto" if torch.cuda.is_available() else "auto"
    ),
    callbacks=callbacks,
    num_sanity_val_steps=0,
    precision=cfg.precision,
    reload_dataloaders_every_n_epochs=5,
    accelerator="gpu",
    devices=cfg.gpus,
    num_nodes=cfg.nodes,
)

# ---------
# INFERENCE
# ---------

session_timestamp = {}
session_subtask_index = {}
session_gt_output = {}
session_pred_output = {}

for batch in tqdm(val_loader):
    absolute_starts = batch.pop("absolute_start")  # (B,)
    session_ids = batch.pop("session_id")  # (B,)
    output_subtask_index = batch.pop("output_subtask_index")

    batch_format = None
    if "input_mask" in batch:
        batch_format = "padded"
    elif "input_seqlen" in batch:
        batch_format = "chained"
    else:
        raise ValueError("Invalid batch format.")

    # move_to_gpu(batch, pl_module)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    move_to_gpu(batch, device)

    # Autocast is explicitly set based on the precision specified by the user.
    # By default, torch autocasts to float16 for 16-bit inference.
    # This behavior is overridden to use bfloat16 if specified in trainer.precision.
    # If 16-bit inference is not enabled, autocast is not used.
    def get_autocast_args(trainer):
        if trainer.precision.startswith("bf16"):
            return torch.bfloat16, True
        elif trainer.precision.startswith("16"):
            return torch.float16, True
        else:
            return None, False

    dtype, enabled = get_autocast_args(trainer)
    # forward pass
    with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
        with torch.inference_mode():
            pred_output, loss, losses_taskwise = model(**batch)

    # we need to get the timestamps, the ground truth values, the task ids as well
    # as the subtask ids. since the batch is padded and chained, this is a bit tricky
    # tldr: this extracts the ground truth in the same format as the model output
    batch_size = len(pred_output)
    # get gt_output and timestamps to be in the same format as pred_output
    timestamps = [{} for _ in range(batch_size)]
    subtask_index = [{} for _ in range(batch_size)]
    gt_output = [{} for _ in range(batch_size)]

    # collect ground truth
    for taskname, spec in model.readout.decoder_specs.items():
        taskid = Decoder.from_string(taskname).value

        # get the mask of tokens that belong to this task
        mask = batch["output_decoder_index"] == taskid

        if not torch.any(mask):
            # there is not a single token for this task, so we skip
            continue

        # we need to distribute the outputs to their respective samples

        if batch_format == "padded":
            token_batch = torch.where(mask)[0]
        elif batch_format == "chained":
            token_batch = batch["output_batch_index"][mask]

        batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
        for i in range(len(batch_i)):
            timestamps[batch_i[i]][taskname] = (
                batch["output_timestamps"][mask][token_batch == i]
                + absolute_starts[batch_i[i]]
            )
            subtask_index[batch_i[i]][taskname] = output_subtask_index[
                taskname
            ][(token_batch == i).detach().cpu()]
            gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                token_batch == i
            ]

    # register all of the data
    for i in range(batch_size):
        session_id = session_ids[i]

        if session_id not in session_pred_output:
            session_pred_output[session_id] = {}
            session_gt_output[session_id] = {}
            session_timestamp[session_id] = {}
            session_subtask_index[session_id] = {}

        for taskname, pred_values in pred_output[i].items():
            if taskname not in session_pred_output[session_id]:
                session_pred_output[session_id][
                    taskname
                ] = pred_values.detach().cpu()
                session_gt_output[session_id][taskname] = (
                    gt_output[i][taskname].detach().cpu()
                )
                session_timestamp[session_id][taskname] = (
                    timestamps[i][taskname].detach().cpu()
                )
                session_subtask_index[session_id][taskname] = (
                    subtask_index[i][taskname].detach().cpu()
                )
            else:
                session_pred_output[session_id][taskname] = torch.cat(
                    (
                        session_pred_output[session_id][taskname],
                        pred_values.detach().cpu(),
                    )
                )
                session_gt_output[session_id][taskname] = torch.cat(
                    (
                        session_gt_output[session_id][taskname],
                        gt_output[i][taskname].detach().cpu(),
                    )
                )
                session_timestamp[session_id][taskname] = torch.cat(
                    (
                        session_timestamp[session_id][taskname],
                        timestamps[i][taskname].detach().cpu(),
                    )
                )
                session_subtask_index[session_id][taskname] = torch.cat(
                    (
                        session_subtask_index[session_id][taskname],
                        subtask_index[i][taskname].detach().cpu(),
                    )
                )

# -----
# EVAL
# -----

results = {"gabors": {}, "static_gratings": {}, "drifting_gratings": {}, "running_speed": {}}

print(dataset.get_session_data(session_ids[0]))

if args.behavior == "gabors":
    gt = dataset.get_session_data(session_ids[0]).gabors.gabors_orientation
    gt = session_gt_output[session_id]['GABOR_ORIENTATION']
    pred = session_pred_output[session_id]['GABOR_ORIENTATION'].argmax(-1)
    results['gabors']['accuracy'] = accuracy_score(gt, pred)
elif args.behavior == "static_gratings":
    gt = dataset.get_session_data(session_ids[0]).static_gratings.static_gratings
    gt = session_gt_output[session_id]['STATIC_GRATINGS']
    pred = session_pred_output[session_id]['STATIC_GRATINGS'].argmax(-1)
    results['static_gratings']['accuracy'] = accuracy_score(gt, pred)
elif args.behavior == "drifting_gratings":
    gt = dataset.get_session_data(session_ids[0]).drifting_gratings.drifting_gratings
    gt = session_gt_output[session_id]['DRIFTING_GRATINGS']
    pred = session_pred_output[session_id]['DRIFTING_GRATINGS'].argmax(-1)
    results['drifting_gratings']['accuracy'] = accuracy_score(gt, pred)
elif args.behavior == "running_speed":
    gt = session_gt_output[session_id]['RUNNING_SPEED']
    pred = session_pred_output[session_id]['RUNNING_SPEED']
    timestamps = session_timestamp[session_id]['RUNNING_SPEED']
    subtask_index = session_subtask_index[session_id]['RUNNING_SPEED']

    gt_vals = gt.squeeze()
    pred_vals = pred.squeeze()

    intervals = np.c_[
        dataset.get_session_data(session_ids[0]).running_speed.start, 
        dataset.get_session_data(session_ids[0]).running_speed.end, 
    ]
    
    _, binned_gt, _, _ = bin_target(
        timestamps,
        gt_vals.numpy(),
        start=intervals.T[0], 
        end=intervals.T[1],
        binsize=params["binsize"], 
        length=sequence_length,
    )
        
    _, binned_pred, _, _ = bin_target(
        timestamps,
        pred_vals.numpy(),
        start=intervals.T[0], 
        end=intervals.T[1],
        binsize=params["binsize"], 
        length=sequence_length,
    )

    results[behavior]["r2_trial"] = r2_score(binned_gt.flatten(), binned_pred.flatten())

    save_res = {
        "gt": binned_gt,
        "pred": binned_pred,
        "beh_name": behavior,
        "session_id": session_id
    }

print(results)

res_path = f"{save_path}/{session_id}/"
if not os.path.exists(res_path):
    os.makedirs(res_path)
np.save(f"{res_path}/{args.behavior}.npy", results)
