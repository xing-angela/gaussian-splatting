import matplotlib.pyplot as plt
import json
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
args = parser.parse_args()

interested_points = list(range(1000, 30001, 1000))
scenes = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118]
for scene in scenes:
    iterations = interested_points
    with open(f'{args.in_dir}/{scene}/log_info.json') as f:
        data = json.load(f)
    num_gaussians = [data["total_gaussians"][i-1] for i in interested_points]
    time_taken = [data["time"][i-1] for i in interested_points]
    
    psnr_values = []
    
    
    for i in interested_points:
        print(f'{args.in_dir}/{scene}/test/ours_{i}/metrics/metrics.json')
        with open(f'{args.in_dir}/{scene}/test/ours_{i}/metrics/metrics.json') as f:
            _data = json.load(f)
            psnr_values.append(_data["PSNR"])
            
    psnr_values = [round(num, 2) for num in psnr_values]
    time_taken = [round(num, 2) for num in time_taken]
    
    interval = len(num_gaussians) // len(psnr_values)
    psnr_iters = [interval*(_+1) for _ in range(len(psnr_values))]
    # Plotting
    fig, ax1 = plt.subplots()
    ax1.set_title(f"{scene} fmb initilization") # change this
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Gaussians', color='tab:blue')
    ax1.plot(interested_points, num_gaussians, color='tab:blue', label='num_gs')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR', color='tab:red')
    ax2.plot(interested_points, psnr_values, color='tab:red', marker='o', label='psnr')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    plt.savefig(f'{args.in_dir}/{scene}/test/num_gaussians_psnr_vs_iteration.png')
    plt.show()
    
    

interested_points = list(range(1000, 30001, 1000))
scenes = [24, 37, 40, 55, 63, 69, 83, 97, 105, 106, 110, 114, 118]
for scene in scenes:
    iterations = interested_points
    with open(f'{args.in_dir}/{scene}/log_info.json') as f:
        data = json.load(f)
    num_gaussians = [data["total_gaussians"][i-1] for i in interested_points]
    time_taken = [data["time"][i-1] for i in interested_points]
    
    psnr_values = []
    
    
    for i in interested_points:
        print(f'{args.in_dir}/{scene}/test/ours_{i}/metrics/metrics.json')
        with open(f'{args.in_dir}/{scene}/test/ours_{i}/metrics/metrics.json') as f:
            _data = json.load(f)
            psnr_values.append(_data["PSNR"])
            
    psnr_values = [round(num, 2) for num in psnr_values]
    time_taken = [round(num, 2) for num in time_taken]
    
    interval = len(num_gaussians) // len(psnr_values)
    psnr_iters = [interval*(_+1) for _ in range(len(psnr_values))]
    # Plotting
    fig, ax1 = plt.subplots()
    ax1.set_title(f"{scene} fmb initilization") # change this
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Gaussians', color='tab:blue')
    ax1.plot(interested_points, num_gaussians, color='tab:blue', label='num_gs')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Time taken in seconds', color='tab:red')
    ax2.plot(interested_points, time_taken, color='tab:red', marker='o', label='psnr')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    plt.savefig(f'{args.in_dir}/{scene}/test/num_gaussians_time_taken_vs_iteration.png')
    plt.show()