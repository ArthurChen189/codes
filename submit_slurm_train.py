#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

SCRIPT = "sbatch train_codes_on_slurm_a40.sh"
CKPT_FOLDER = "/checkpoint/arthur/"
DATASET_FOLDER = "./data"
def execute_cmd(cmd, ckpt_path, require_confirmation=True):
    if require_confirmation:
        response = input("Do you want to submit the job? (y/n)\n")
        if response.lower() != "y":
            return
    res = os.system(cmd)
    if int(res) != 0:
        print(f"Job failed for {cmd}")
    else:
        Path(ckpt_path, "TAKEN").touch()
 

def submit_job(ckpt_id, dataset_name, ckpt_path, require_confirmation):
    cmd = SCRIPT
    cmd += f" {ckpt_id}"
    cmd += f" {dataset_name}"

    print(f"Submitting job with command:\n{cmd}\n")
    execute_cmd(cmd, ckpt_path, require_confirmation)
    print()


def get_ckpt_ids(num_ckpts):
    ckpt_path = Path(CKPT_FOLDER)
    ckpts = []
    ckpt_paths = []
    counter = num_ckpts
    for ckpt_path in list(ckpt_path.glob("*")):
        if ckpt_path.is_dir() and ckpt_path.stat().st_size <= 1: # 1 means there is only one placeholder file in the folder
            if Path(ckpt_path, "TAKEN").exists():
                continue

            # delete the placeholder file
            Path(ckpt_path, "SLURM_JOB_FINISHED").unlink(missing_ok=True)

            ckpts.append(ckpt_path.name)
            ckpt_paths.append(ckpt_path)
            counter -= 1
            if counter == 0:
                break
    return ckpts, ckpt_paths


def main(args):
    datasets = set()
    dataset_paths = Path(DATASET_FOLDER).glob(args.dataset_regex)
    for path in dataset_paths:
        dataset_name = path.stem.replace("_text2sql", "").split("v1_")[0]
        datasets.add(dataset_name + "v1")

    print(datasets)
    ckpt_ids, ckpt_paths = get_ckpt_ids(len(datasets))
    for dataset, ckpt_id, ckpt_path in zip(datasets, ckpt_ids, ckpt_paths):
        submit_job(ckpt_id, dataset, ckpt_path, args.require_confirmation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train_codes_on_slurm_a40.sh args
    parser.add_argument("--db_id", type=str, choices=["all"], default="all") #TODO: add choices

    # other args
    parser.add_argument("--dataset_regex", type=str, required=True, help="regex to match dataset names")
    parser.add_argument("--require_confirmation", type=bool, default=True)
    args = parser.parse_args()
    main(args)