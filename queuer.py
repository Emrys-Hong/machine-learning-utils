# Disclaimer: Do not hold GPU memory for a long time
# Source:
# - https://github.com/fbcotter/py3nvml
# - https://docs.fast.ai/dev/gpu.html#accessing-nvidia-gpu-info-programmatically
from py3nvml.py3nvml import *
from pynvml import *
import py3nvml
import nvidia_smi
from time import sleep
import torch
import argparse

def check_gpu_stat():
    nvidia_smi.nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu{i}: {res.gpu}%, gpu-mem: {res.memory}%')


def queue(n_request=1, fraction=0.95, gpu_select=None):
    while py3nvml.grab_gpus(num_gpus=n_request, gpu_fraction=fraction, gpu_select=gpu_select) < n_request:
        sleep(15)
    print('Exit Queue')

def check_mem(cuda_device):
    handle = nvmlDeviceGetHandleByIndex(cuda_device)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.total>>20, info.used>>20, info.free>>20

def reserve(h=24):
    nvidia_smi.nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    free_mem = []
    for i in range(deviceCount):
        total, used, free = check_mem(i)
        free_mem.append(free)
    block_mem = int(max(free_mem) * 0.9)
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    try:                      sleep(h*3600)
    except KeyboardInterrupt: print('\nMemory Released')
    del x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--fraction', type=float, default=0.95)
    parser.add_argument('--n_request', type=int, default=1)
    parser.add_argument('--time', type=int, default=24)
    args = parser.parse_args()
    check_gpu_stat()
    queue(args.n_request, args.fraction, args.gpu_id)
    reserve(args.time)

