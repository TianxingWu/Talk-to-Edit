import numpy as np
import torch
N = 50000

torch.manual_seed(0)

random_codes = torch.randn(N, 1, 512)
random_codes = random_codes.numpy()

np.save(f'./download/editing_data/random_codes_{N}.npy', random_codes)

