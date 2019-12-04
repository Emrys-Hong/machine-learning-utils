"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
from multiprocessing import Process
import utils
import torch


gpu_option = [2]
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', required=True, help='Directory containing params.json')
parser.add_argument('--start_id', required=True, help='the number when the epoch starts')
parser.add_argument('--start_cuda', default=0, help='which cuda id to start from')
parser.add_argument('--num_process', default=3, help='how many number of process to start at the same time')
cuda_number = torch.cuda.device_count()

def launch_training_job(main_dir, config_id, cuda_id):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    if not os.path.exists(main_dir):
        raise Exception(model_dir + ' is not created')

    # Launch training with this config
    cmd = "CUDA_VISIBLE_DEVICES={cuda_id} {python} main.py --config_dir={main_dir} --config_id={config_id}"\
            .format(cuda_id=cuda_id, python=PYTHON, main_dir=main_dir, config_id=config_id)
    print(cmd)
    check_call(cmd, shell=True)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    config_dir = os.path.join(args.main_dir, 'configs')
    print(os.listdir(config_dir))
    tasks = []
    for config in os.listdir(config_dir):
        config_id = config.split('_')[-1].split('.')[0]
        if int(config_id) >= int(args.start_id):
            # p = Process(target=launch_training_job, args=(args.main_dir, config_id, len(tasks)%cuda_number))
            p = Process(target=launch_training_job, args=(args.main_dir, config_id,\
                                                          gpu_option[len(tasks)%len(gpu_option)]))
            tasks.append(p)
            p.start()
        if len(tasks) == args.num_process:
            for p in tasks:
                p.join()