#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def execute_cmd(cmd, require_confirmation=True):
    if require_confirmation:
        response = input("Do you want to submit the job? (y/n)\n")
        if response.lower() != "y":
            return
    res = os.system(cmd)
    if int(res) != 0:
        print(f"Job failed for {cmd}")
 

def submit_job(script: str, ckpt_paths: str, output_dirs: str, db_ids: str, require_confirmation: bool):
    cmd = script
    cmd += f' \"{ckpt_paths}\"'
    cmd += f' \"{output_dirs}\"'
    cmd += f' \"{db_ids}\"'

    print(f"Submitting job with command:\n{cmd}\n")
    execute_cmd(cmd, require_confirmation)
    print()


def main(args):
    # ---------------- constants ----------------
    SCRIPT = "sbatch eval_on_slurm_a40.sh"
    CKPT_FOLDER = f"/projects/r2llab/arthur/checkpoints/Text2SQL"
    OUTPUT_FOLDER = "./qwen_1.5b_instruct_predictions/data_synthesized_by_qwen2.5-coder-7b-instruct/"

    # ckpt_maps = {
    #     "llama3.1-8b_1000_zero-shot_bg_v1": 
    #         {"computer_student": "14972740", "movie_platform": "14972740", "app_store": "14988145"},
    #     "llama3.1-8b_1000_zero-shot_bg_test-time-info_v1":
    #         {"computer_student": "14972843", "movie_platform": "14972843", "app_store": "14988141"},
    #     "llama3.1-8b_1000_few-shot_v1":
    #         {"computer_student": "14972737", "movie_platform": "14972737", "app_store": "14987858"},
    #     "llama3.1-8b_1000_few-shot_bg_v1":
    #         {"computer_student": "14972726", "movie_platform": "14972726", "app_store": "14987868"},
    #     "llama3.1-8b_1000_few-shot_bg_test-time-info_v1":
    #         {"computer_student": "14991452", "movie_platform": "14991452", "app_store": "14987870"},
    # }

    ckpt_maps = {
        "qwen2.5-coder-7b_1000_few-shot_bg_test-time-info_v1":
            {"computer_student": "computer_student", "movie_platform": "movie_platform", "app_store": "app_store"},
        "qwen2.5-coder-7b_1000_few-shot_bg_v1":
            {"computer_student": "computer_student", "movie_platform": "movie_platform", "app_store": "app_store"},
        "qwen2.5-coder-7b_1000_zero-shot_bg_test-time-info_v1":
            {"computer_student": "computer_student", "movie_platform": "movie_platform", "app_store": "app_store"},
        "qwen2.5-coder-7b_1000_zero-shot_bg_v1":
            {"computer_student": "computer_student", "movie_platform": "movie_platform", "app_store": "app_store"}
    }
    # ---------------- end of constants ----------------

    if args.n_th_recent_ckpt == 1:
        OUTPUT_FOLDER = OUTPUT_FOLDER + "last_checkpoint"
    elif args.n_th_recent_ckpt == 2:
        OUTPUT_FOLDER = OUTPUT_FOLDER + "second_last_checkpoint"
    elif args.n_th_recent_ckpt == 3:
        OUTPUT_FOLDER = OUTPUT_FOLDER + "third_last_checkpoint"
    else:
        raise NotImplementedError(f"Invalid n_th_recent_ckpt: {args.n_th_recent_ckpt}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    db_ckpt_id_map = ckpt_maps[args.dataset_name]
    ckpt_paths = []
    output_dirs = []
    db_ids = []
    for db_id, ckpt_id in db_ckpt_id_map.items():
        ckpt_path = Path(CKPT_FOLDER, args.dataset_name, f"{ckpt_id}")
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path} does not exist")
            raise ValueError(f"Checkpoint {ckpt_path} does not exist")
        else:
            candidate_paths = list(ckpt_path.glob(f"ckpt*"))
            sorted_paths = sorted(candidate_paths, key=lambda x: int(x.stem.split("-")[-1]), reverse=True)
            final_ckpt_path = sorted_paths[args.n_th_recent_ckpt - 1]
            id_num = final_ckpt_path.stem.split("-")[-1]
            output_path = Path(OUTPUT_FOLDER, args.dataset_name + "_" + db_id + f"-ckpt-{id_num}.json")

            ckpt_paths.append(str(final_ckpt_path))
            output_dirs.append(str(output_path))
            db_ids.append(db_id)

    submit_job(SCRIPT, " ".join(ckpt_paths), " ".join(output_dirs), " ".join(db_ids), args.require_confirmation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train_codes_on_slurm_a40.sh args
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--n_th_recent_ckpt", type=int, default=1, help="1 means the latest ckpt, 2 means the second latest ckpt, etc.")

    # other args
    parser.add_argument("--require_confirmation", type=bool, default=True)
    args = parser.parse_args()
    main(args)