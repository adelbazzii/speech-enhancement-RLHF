import copy
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.cmgan.src.data.dataloader import DemandDataset
from models.cmgan.src.models.generator import TSCNet as CMGANGenerator
from models.metricgan_plus.model import Generator as MetricGANGenerator


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cmgan(ckpt_path, device):
    generator = CMGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
    else:
        print("No CMGAN checkpoint found. Starting training from scratch.")

    return generator


def load_metricgan(ckpt_path, device):
    generator = MetricGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
    else:
        print("Np MetricGAN+ checkpoint found. Starting training from scratch.")

    return generator


def create_dataloader(data_dir, split, batch_size, num_workers):
    ds = DemandDataset(os.path.join(data_dir, split))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dl


def train_cmgan_rlhf(device, lr, data_dir, batch_size, num_workers, max_steps, accum_steps):
    # rl poliscy
    rl_generator = load_cmgan(device=device)
    rl_generator.train()

    # sft policy
    sft_generator = copy.deepcopy(rl_generator)
    sft_generator.eval()
    for p in sft_generator.parameters():
        p.requires_grad = False
    
    # optimizer
    optimizer = torch.optim.Adam(rl_generator.parameters(), lr=lr)

    # data loader
    dataloader = create_dataloader(
        data_dir=data_dir,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_iter = iter(dataloader)
    for step in tqdm(range(1, max_steps+1), desc="CMGAN RLHF"):
        optimizer.zero_grad()

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            clean = batch[0].to(device)
            noisy = batch[1].to(device)

            # sft policy
            with torch.no_grad():

            
            # rl policy



def train_metrocgan_rlhf(device):
    rl_generator = load_metricgan(device=device)


def main():
    device = get_device()
