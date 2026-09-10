from pytorch_fid.fid_score import calculate_fid_given_paths
import torch
import json
from test_script import ddim_uncond_sample

with open('config.json', 'r') as f:
    config = json.load(f)

len_dataset = len(os.listdir(config['data_fp']))
samples = ddim_uncond_sample(net, dm, config, num_samples = len_dataset)

fid = calculate_fid_given_paths(
    ["dataset/real", "gen_samples"],
    batch_size=10,
    device="cuda",
    dims=2048,
    num_workers=4
)

print(f'FID {fid}')